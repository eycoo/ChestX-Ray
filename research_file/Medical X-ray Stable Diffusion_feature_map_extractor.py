"""Zero-disk Medical SD extractor with integrated FA pipeline.

Pipeline (single file, no raw .pt write/read):
1) Encode image to latent.
2) Add diffusion noise at selected timestep.
3) Build text conditioning and run one U-Net forward.
4) Capture feature maps + attention maps directly from memory.
5) Run FA model and return compact latent vector (z).
6) Save only final .npy vector.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time

try:
    from diffusers import StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - import guard for clearer runtime message
    raise SystemExit(
        "diffusers belum terpasang. Install dulu: pip install diffusers transformers accelerate"
    ) from exc


TensorDict = Dict[str, torch.Tensor]


def _first_tensor(value: object) -> Optional[torch.Tensor]:
    """Return first tensor found inside nested outputs (tensor/tuple/list/dict)."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


@dataclass
class ExtractionConfig:
    model_id: str
    base_model_id: Optional[str]
    lora_weight_name: Optional[str]
    image_path: Path
    prompt: str
    output_dir: Path
    timestep: int
    blocks: str
    dtype: str
    width: int
    height: int
    device: str
    local_files_only: bool


class BlockFeatureCollector:
    """Collect 4D feature maps from chosen U-Net blocks via forward hooks."""

    def __init__(self, unet: nn.Module, blocks: str):
        self.unet = unet
        self.blocks = blocks
        self.features: TensorDict = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _args: tuple, output: object) -> None:
            tensor = _first_tensor(output)
            if tensor is not None and tensor.dim() == 4:
                self.features[name] = tensor.detach()

        return _hook

    def register(self) -> None:
        for idx, block in enumerate(self.unet.up_blocks):
            self._hooks.append(block.register_forward_hook(self._make_hook(f"up_{idx}")))

        if self.blocks in {"up_mid", "all"} and hasattr(self.unet, "mid_block"):
            self._hooks.append(self.unet.mid_block.register_forward_hook(self._make_hook("mid_0")))

        if self.blocks == "all":
            for idx, block in enumerate(self.unet.down_blocks):
                self._hooks.append(block.register_forward_hook(self._make_hook(f"down_{idx}")))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class CaptureCrossAttentionProcessor(nn.Module):
    """Drop-in attention processor that stores cross-attention as spatial maps."""

    def __init__(self, name: str, store: TensorDict):
        super().__init__()
        self.name = name
        self.store = store

    def _save_spatial_attention(
        self,
        attention_probs: torch.Tensor,
        batch_size: int,
        spatial_h: Optional[int],
        spatial_w: Optional[int],
    ) -> None:
        heads = attention_probs.shape[0] // batch_size
        q_tokens = attention_probs.shape[1]
        k_tokens = attention_probs.shape[2]
        probs = attention_probs.view(batch_size, heads, q_tokens, k_tokens)

        saliency = probs.mean(dim=1).mean(dim=-1)  # [B, q_tokens]

        if spatial_h is None or spatial_w is None:
            side = int(math.sqrt(q_tokens))
            if side * side == q_tokens:
                spatial_h, spatial_w = side, side
            else:
                spatial_h, spatial_w = 1, q_tokens

        saliency_map = saliency.view(batch_size, 1, spatial_h, spatial_w)
        self.store[self.name] = saliency_map.detach()

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross_attention = encoder_hidden_states is not None

        spatial_h: Optional[int] = None
        spatial_w: Optional[int] = None
        if input_ndim == 4:
            batch_size, channel, spatial_h, spatial_w = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, spatial_h * spatial_w).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attention:
            self._save_spatial_attention(attention_probs, batch_size, spatial_h, spatial_w)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, spatial_h, spatial_w)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttentionCollector:
    """Install custom processors on U-Net attention modules and collect maps."""

    def __init__(self, unet: nn.Module):
        self.unet = unet
        self.maps: TensorDict = {}
        self._original_processors = None

    def install(self) -> None:
        self._original_processors = self.unet.attn_processors
        custom_processors = {
            name: CaptureCrossAttentionProcessor(name=name, store=self.maps)
            for name in self.unet.attn_processors.keys()
        }
        self.unet.set_attn_processor(custom_processors)

    def restore(self) -> None:
        if self._original_processors is not None:
            self.unet.set_attn_processor(self._original_processors)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def _load_pipeline(config: ExtractionConfig) -> Tuple[StableDiffusionPipeline, torch.dtype]:
    dtype = _resolve_dtype(config.dtype)

    if config.base_model_id:
        pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=dtype,
            local_files_only=config.local_files_only,
        )
        load_kwargs = {
            "weight_name": config.lora_weight_name,
            "local_files_only": config.local_files_only,
        }
        try:
            pipe.load_lora_weights(config.model_id, **load_kwargs)
        except TypeError:
            load_kwargs.pop("local_files_only", None)
            pipe.load_lora_weights(config.model_id, **load_kwargs)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            local_files_only=config.local_files_only,
        )

    pipe = pipe.to(config.device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe, dtype


def _preprocess_image(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, H, W]
    image_tensor = torch.from_numpy(arr)
    image_tensor = image_tensor * 2.0 - 1.0
    return image_tensor


def _encode_latent(
    pipe: StableDiffusionPipeline,
    image_tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_tensor = image_tensor.to(pipe.device, dtype=dtype)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(image_tensor).latent_dist
        latents = latent_dist.sample() * pipe.vae.config.scaling_factor
    return latents


def _build_text_embeddings(pipe: StableDiffusionPipeline, prompt: str) -> torch.Tensor:
    tokens = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(pipe.device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(input_ids)[0]
    return text_embeddings


def _add_noise_at_timestep(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    timestep: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    t = torch.full((latents.shape[0],), timestep, dtype=torch.long, device=latents.device)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
    return noisy_latents, t


class SpatialAttention(nn.Module):
    """Scaled dot-product spatial attention with optional downsampling for stability."""

    def __init__(self, in_channels: int, num_heads: int = 4, max_hw: int = 32):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = max(in_channels // self.num_heads, 1)
        self.inner_dim = self.head_dim * self.num_heads
        self.max_hw = max_hw

        self.q = nn.Conv2d(in_channels, self.inner_dim, 1, bias=False)
        self.k = nn.Conv2d(in_channels, self.inner_dim, 1, bias=False)
        self.v = nn.Conv2d(in_channels, self.inner_dim, 1, bias=False)
        self.proj = nn.Conv2d(self.inner_dim, in_channels, 1, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_attn = x
        resized = False

        if self.max_hw > 0 and (h > self.max_hw or w > self.max_hw):
            x_attn = F.interpolate(x, size=(self.max_hw, self.max_hw), mode="bilinear", align_corners=False)
            resized = True

        b2, _, h2, w2 = x_attn.shape
        n = h2 * w2

        q = self.q(x_attn).reshape(b2, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = self.k(x_attn).reshape(b2, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = self.v(x_attn).reshape(b2, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(b2, self.inner_dim, h2, w2)
        out = self.proj(out)

        if resized:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class DFATBLayer(nn.Module):
    """Single-scale DFATB layer: spatial + channel attention fusion."""

    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 4, spatial_max_hw: int = 32):
        super().__init__()
        self.spatial = SpatialAttention(in_channels, num_heads=num_heads, max_hw=spatial_max_hw)
        self.channel = ChannelAttention(in_channels, reduction=reduction)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial(x) + self.channel(x)
        return self.norm(out)


class DFATB(nn.Module):
    """Multi-scale DFATB over feature list."""

    def __init__(self, channel_list: Sequence[int], num_heads: int = 4, reduction: int = 4, spatial_max_hw: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DFATBLayer(c, num_heads=num_heads, reduction=reduction, spatial_max_hw=spatial_max_hw)
                for c in channel_list
            ]
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(fm) for layer, fm in zip(self.layers, feature_maps)]


class FAFNLayer(nn.Module):
    """Single-scale FAFN with split-gate and 1x1 MLP."""

    def __init__(self, in_channels: int, expansion: int = 2):
        super().__init__()
        half = max(in_channels // 2, 1)
        mid = half * max(expansion, 1)
        self.half = half

        self.w1 = nn.Conv2d(half, mid, 1, bias=False)
        self.w2 = nn.Conv2d(mid, in_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = x[:, : self.half, :, :]
        f2 = x[:, self.half : self.half * 2, :, :]
        if f2.shape[1] != f1.shape[1]:
            f2 = F.pad(f2, (0, 0, 0, 0, 0, f1.shape[1] - f2.shape[1]))

        f_gate = f1 * torch.sigmoid(f2)
        f_fafn = self.w2(F.relu(self.w1(f_gate)))
        return self.norm(f_fafn)


class FAFN(nn.Module):
    """Multi-scale FAFN over feature list."""

    def __init__(self, channel_list: Sequence[int], expansion: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([FAFNLayer(c, expansion=expansion) for c in channel_list])

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(fm) for layer, fm in zip(self.layers, feature_maps)]


class DifferentialDenoising(nn.Module):
    """Differential denoising with learnable lambda per feature scale."""

    def __init__(self, feature_channels: Sequence[int], attn_channels: Sequence[int], lambda_init: float = 0.5):
        super().__init__()
        self.lambdas = nn.ParameterList(
            [nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32)) for _ in feature_channels]
        )
        self.attn_proj = nn.ModuleList(
            [nn.Conv2d(ac, fc, 1, bias=False) for fc, ac in zip(feature_channels, attn_channels)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm2d(fc) for fc in feature_channels])

    def forward(
        self,
        ffafn_list: List[torch.Tensor],
        attn_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        fdenoise_list: List[torch.Tensor] = []
        adiff_list: List[torch.Tensor] = []

        for i, (ffafn, (a1, a2)) in enumerate(zip(ffafn_list, attn_pairs)):
            lam = torch.sigmoid(self.lambdas[i])

            if a2.shape[-2:] != a1.shape[-2:]:
                a2 = F.interpolate(a2, size=a1.shape[-2:], mode="bilinear", align_corners=False)

            if a1.shape[1] != a2.shape[1]:
                if a1.shape[1] == 1:
                    a2 = a2.mean(dim=1, keepdim=True)
                elif a2.shape[1] == 1:
                    a2 = a2.expand(-1, a1.shape[1], -1, -1)
                else:
                    min_c = min(a1.shape[1], a2.shape[1])
                    a1 = a1[:, :min_c, :, :]
                    a2 = a2[:, :min_c, :, :]

            adiff = a1 - lam * a2
            adiff_proj = self.attn_proj[i](adiff)

            if adiff_proj.shape[-2:] != ffafn.shape[-2:]:
                adiff_proj = F.interpolate(adiff_proj, size=ffafn.shape[-2:], mode="bilinear", align_corners=False)

            mask = torch.sigmoid(adiff_proj)
            fdenoise = self.norms[i](ffafn * mask)

            fdenoise_list.append(fdenoise)
            adiff_list.append(adiff)

        return fdenoise_list, adiff_list


class GAPConcat(nn.Module):
    """GAP each feature/attention then concatenate."""

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, fdenoise_list: List[torch.Tensor], adiff_list: List[torch.Tensor]) -> torch.Tensor:
        vj = [self.gap(f).flatten(1) for f in fdenoise_list]
        ul = [self.gap(a).flatten(1) for a in adiff_list]
        return torch.cat(vj + ul, dim=1)


class BottleneckProjection(nn.Module):
    """Projection from fused vector to compact latent space."""

    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        return self.proj(z_fused)


class FeatureAggregationEncoder(nn.Module):
    """End-to-end FA encoder that outputs latent z vector."""

    def __init__(
        self,
        feature_channels: Sequence[int],
        attn_channels: Sequence[int],
        z_dim: int = 128,
        num_heads: int = 4,
        fafn_expansion: int = 2,
        lambda_init: float = 0.5,
        dropout: float = 0.1,
        spatial_max_hw: int = 32,
    ):
        super().__init__()
        self.dfatb = DFATB(feature_channels, num_heads=num_heads, spatial_max_hw=spatial_max_hw)
        self.fafn = FAFN(feature_channels, expansion=fafn_expansion)
        self.diff_denoise = DifferentialDenoising(feature_channels, attn_channels, lambda_init=lambda_init)
        self.gap_concat = GAPConcat()

        c_tot = sum(feature_channels) + sum(attn_channels)
        self.bottleneck = BottleneckProjection(c_tot, out_dim=z_dim, dropout=dropout)

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        attn_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        f_star = self.dfatb(feature_maps)
        f_fafn = self.fafn(f_star)
        f_denoise, a_diff = self.diff_denoise(f_fafn, attn_pairs)
        z_fused = self.gap_concat(f_denoise, a_diff)
        z = self.bottleneck(z_fused)
        return {"z": z, "z_fused": z_fused}


def _prepare_sorted_maps(
    feature_maps: TensorDict,
    attention_maps: TensorDict,
) -> Tuple[List[str], List[torch.Tensor], List[str], List[torch.Tensor]]:
    feature_items = [(k, v.float()) for k, v in sorted(feature_maps.items()) if isinstance(v, torch.Tensor) and v.dim() == 4]
    if not feature_items:
        raise RuntimeError("Tidak ada feature maps 4D untuk FA pipeline.")

    attn_items = [(k, v.float()) for k, v in sorted(attention_maps.items()) if isinstance(v, torch.Tensor) and v.dim() == 4]

    feature_keys = [k for k, _ in feature_items]
    feature_list = [v for _, v in feature_items]

    if not attn_items:
        attn_items = []
        for idx, f in enumerate(feature_list):
            attn_items.append((f"synthetic_attn_{idx}", f.mean(dim=1, keepdim=True)))

    attention_keys = [k for k, _ in attn_items]
    attention_list = [v for _, v in attn_items]

    return feature_keys, feature_list, attention_keys, attention_list


def _build_attention_pairs(
    feature_list: List[torch.Tensor],
    attention_keys: List[str],
    attention_list: List[torch.Tensor],
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[int], List[Tuple[str, str]]]:
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    attn_channels: List[int] = []
    pair_names: List[Tuple[str, str]] = []

    attn_areas = [a.shape[-2] * a.shape[-1] for a in attention_list]

    for f in feature_list:
        target_area = f.shape[-2] * f.shape[-1]
        ranked_idx = sorted(
            range(len(attention_list)),
            key=lambda i: abs(attn_areas[i] - target_area),
        )

        i1 = ranked_idx[0]
        i2 = ranked_idx[1] if len(ranked_idx) > 1 else ranked_idx[0]

        a1 = attention_list[i1]
        a2 = attention_list[i2]

        pairs.append((a1, a2))
        attn_channels.append(a1.shape[1])
        pair_names.append((attention_keys[i1], attention_keys[i2]))

    return pairs, attn_channels, pair_names


def _model_feature_channels(fa_model: FeatureAggregationEncoder) -> List[int]:
    return [layer.spatial.q.in_channels for layer in fa_model.dfatb.layers]


def _model_attention_channels(fa_model: FeatureAggregationEncoder) -> List[int]:
    return [proj.in_channels for proj in fa_model.diff_denoise.attn_proj]


def _align_features_to_model(
    feature_keys: List[str],
    feature_list: List[torch.Tensor],
    expected_channels: List[int],
) -> Tuple[List[str], List[torch.Tensor]]:
    if len(feature_list) < len(expected_channels):
        raise RuntimeError(
            f"Jumlah feature map ({len(feature_list)}) lebih kecil dari kebutuhan model FA ({len(expected_channels)})."
        )

    selected_keys: List[str] = []
    selected_maps: List[torch.Tensor] = []
    used_indices: set[int] = set()

    for target_c in expected_channels:
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[int, int]] = None

        for idx, fmap in enumerate(feature_list):
            if idx in used_indices:
                continue
            channel_gap = abs(fmap.shape[1] - target_c)
            area_score = -(fmap.shape[-2] * fmap.shape[-1])
            score = (channel_gap, area_score)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            raise RuntimeError("Gagal memilih feature map untuk seluruh skala FA.")

        used_indices.add(best_idx)
        selected_keys.append(feature_keys[best_idx])
        selected_maps.append(feature_list[best_idx])

    return selected_keys, selected_maps


def _adapt_attention_channels(attn: torch.Tensor, target_channels: int) -> torch.Tensor:
    current_channels = attn.shape[1]
    if current_channels == target_channels:
        return attn
    if target_channels <= 0:
        raise ValueError("target_channels harus > 0.")

    if current_channels == 1 and target_channels > 1:
        return attn.expand(-1, target_channels, -1, -1)
    if current_channels > target_channels:
        return attn[:, :target_channels, :, :]

    repeat_factor = (target_channels + current_channels - 1) // current_channels
    expanded = attn.repeat(1, repeat_factor, 1, 1)
    return expanded[:, :target_channels, :, :]


def run_extraction_with_pipeline(
    pipe: StableDiffusionPipeline,
    dtype: torch.dtype,
    config: ExtractionConfig,
    fa_model: FeatureAggregationEncoder,
    feature_collector: BlockFeatureCollector, # <--- TAMBAH INI
    attn_collector: AttentionCollector,       # <--- TAMBAH INI
) -> np.ndarray:
    """Run extraction for a single image using an already-loaded pipeline."""
# HAPUS BARIS INI:
    # feature_collector = BlockFeatureCollector(pipe.unet, blocks=config.blocks)
    # attn_collector = AttentionCollector(pipe.unet)
    # feature_collector.register()
    # attn_collector.install()

    try:
        image_tensor = _preprocess_image(config.image_path, config.width, config.height)
        latents = _encode_latent(pipe, image_tensor=image_tensor, dtype=dtype)
        text_embeddings = _build_text_embeddings(pipe, prompt=config.prompt)
        noisy_latents, timestep = _add_noise_at_timestep(pipe, latents=latents, timestep=config.timestep)

        with torch.no_grad():
            _ = pipe.unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=True,
            )

        feature_maps = {k: v for k, v in feature_collector.features.items() if v.dim() == 4}
        attention_maps = {k: v for k, v in attn_collector.maps.items() if v.dim() == 4}

        if not feature_maps:
            raise RuntimeError("Tidak ada feature map 4D yang tertangkap.")

        feature_keys, feature_list, attention_keys, attention_list = _prepare_sorted_maps(feature_maps, attention_maps)
        expected_feature_channels = _model_feature_channels(fa_model)
        selected_feature_keys, selected_feature_list = _align_features_to_model(
            feature_keys, feature_list, expected_feature_channels,
        )
        attn_pairs, _, pair_names = _build_attention_pairs(
            selected_feature_list, attention_keys, attention_list,
        )
        expected_attn_channels = _model_attention_channels(fa_model)
        aligned_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for (a1, a2), target_attn_c in zip(attn_pairs, expected_attn_channels):
            aligned_pairs.append((
                _adapt_attention_channels(a1, target_attn_c),
                _adapt_attention_channels(a2, target_attn_c),
            ))

        with torch.no_grad():
            f_dev = [f.to(config.device) for f in selected_feature_list]
            a_dev = [(a1.to(config.device), a2.to(config.device)) for (a1, a2) in aligned_pairs]
            fa_outputs = fa_model(f_dev, a_dev)

        z_vector = fa_outputs["z"].detach().float().cpu().numpy()
        return z_vector

    finally:
        # Bersihkan kamusnya saja, JANGAN cabut hook-nya
        feature_collector.features.clear()
        attn_collector.maps.clear()
        
        # HAPUS/COMMENT 2 baris ini:
        # feature_collector.remove()
        # attn_collector.restore()
        
        # Per-image VRAM flush
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_integrated_pipeline(config: ExtractionConfig, fa_model: FeatureAggregationEncoder) -> np.ndarray:
    """Legacy wrapper: loads pipeline internally. Use run_extraction_with_pipeline for batch."""
    pipe, dtype = _load_pipeline(config)
    try:
        return run_extraction_with_pipeline(pipe, dtype, config, fa_model)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_int_list(raw: str, arg_name: str) -> List[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Argumen {arg_name} kosong.")
    parsed: List[int] = []
    for val in values:
        parsed.append(int(val))
    return parsed


def _collect_jobs_from_csv(
    prompts_csv: Path,
    image_col: str,
    prompt_col: str,
    image_dir: Optional[Path],
    max_samples: int,
) -> List[Tuple[Path, str]]:
    jobs: List[Tuple[Path, str]] = []

    with prompts_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if image_col not in fieldnames:
            raise ValueError(f"Kolom image tidak ditemukan: {image_col}")
        if prompt_col not in fieldnames:
            raise ValueError(f"Kolom prompt tidak ditemukan: {prompt_col}")

        for row in reader:
            image_name = (row.get(image_col) or "").strip()
            prompt = (row.get(prompt_col) or "").strip()

            if not image_name or not prompt:
                continue

            image_path = Path(image_name)
            if not image_path.is_absolute():
                if image_dir is not None:
                    image_path = image_dir / image_path
                else:
                    image_path = prompts_csv.parent / image_path

            if not image_path.exists():
                continue

            jobs.append((image_path, prompt))

            if max_samples > 0 and len(jobs) >= max_samples:
                break

    if not jobs:
        raise ValueError("Tidak ada job valid dari CSV.")

    return jobs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-disk Medical SD extractor + FA pipeline end-to-end.",
    )

    # Single item mode
    parser.add_argument("--image-path", default=None, help="Path image input (mode single sample)")
    parser.add_argument("--prompt", default=None, help="Prompt teks conditioning (mode single sample)")

    # Dataset mode
    parser.add_argument("--prompts-csv", default=None, help="CSV prompts untuk mode batch")
    parser.add_argument("--image-dir", default=None, help="Root folder gambar untuk mode batch")
    parser.add_argument("--image-col", default="Image Index", help="Nama kolom image di CSV")
    parser.add_argument("--prompt-col", default="Prompt", help="Nama kolom prompt di CSV")
    parser.add_argument("--max-samples", type=int, default=0, help="Batas jumlah sample di mode batch (0 = semua)")

    # Extraction config
    parser.add_argument("--model-id", required=True, help="Model path atau model id Hugging Face")
    parser.add_argument(
        "--base-model-id",
        default=None,
        help=(
            "Base model SD jika --model-id adalah repo LoRA adapter. "
            "Contoh: CompVis/stable-diffusion-v1-4"
        ),
    )
    parser.add_argument(
        "--lora-weight-name",
        default="pytorch_lora_weights.safetensors",
        help="Nama file LoRA weight di repo adapter",
    )
    parser.add_argument("--output-dir", default="outputs/fe_with_fa", help="Folder output final .npy")
    parser.add_argument("--timestep", type=int, default=10, help="Timestep diffusion untuk noising")
    parser.add_argument(
        "--blocks",
        choices=["up", "up_mid", "all"],
        default="up_mid",
        help="Blok U-Net yang dihook untuk feature map",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision saat inferensi",
    )
    parser.add_argument("--width", type=int, default=512, help="Lebar resize input")
    parser.add_argument("--height", type=int, default=512, help="Tinggi resize input")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device inferensi",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Aktifkan jika model hanya tersedia secara lokal",
    )

    # FA model config
    parser.add_argument("--fa-z-dim", type=int, default=128, help="Dimensi output FA bottleneck")
    parser.add_argument("--fa-num-heads", type=int, default=4, help="Jumlah head spatial attention FA")
    parser.add_argument("--fa-fafn-expansion", type=int, default=2, help="Expansion factor FAFN")
    parser.add_argument("--fa-lambda-init", type=float, default=0.5, help="Nilai awal lambda differential denoising")
    parser.add_argument("--fa-dropout", type=float, default=0.1, help="Dropout di bottleneck FA")
    parser.add_argument(
        "--fa-spatial-max-hw",
        type=int,
        default=32,
        help="Batas resolusi (H/W) sebelum spatial attention FA melakukan downsample",
    )
    parser.add_argument(
        "--fa-feature-channels",
        default="320,640,1280,1280",
        help="Daftar channel feature untuk model FA, dipisah koma",
    )
    parser.add_argument(
        "--fa-attn-channels",
        default="1,1,1,1",
        help="Daftar channel attention untuk model FA, dipisah koma",
    )
    parser.add_argument(
        "--fa-checkpoint",
        default=None,
        help="Path checkpoint FA (state_dict atau dict dengan key model_state)",
    )
    parser.add_argument(
        "--save-fa-details",
        action="store_true",
        help="Deprecated: argumen ini diabaikan pada mode zero-disk.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.save_fa_details:
        print("Info: --save-fa-details diabaikan pada mode zero-disk.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_channels = _parse_int_list(args.fa_feature_channels, "--fa-feature-channels")
    attn_channels = _parse_int_list(args.fa_attn_channels, "--fa-attn-channels")
    if len(feature_channels) != len(attn_channels):
        raise ValueError("Panjang --fa-feature-channels harus sama dengan --fa-attn-channels.")

    # 1) Setup FA model sekali di awal.
    fa_model = FeatureAggregationEncoder(
        feature_channels=feature_channels,
        attn_channels=attn_channels,
        z_dim=args.fa_z_dim,
        num_heads=args.fa_num_heads,
        fafn_expansion=args.fa_fafn_expansion,
        lambda_init=args.fa_lambda_init,
        dropout=args.fa_dropout,
        spatial_max_hw=args.fa_spatial_max_hw,
    ).to(args.device)

    if args.fa_checkpoint and Path(args.fa_checkpoint).exists():
        state = torch.load(args.fa_checkpoint, map_location=args.device)
        if isinstance(state, dict):
            if "fa_model_state" in state:
                state = state["fa_model_state"]
            elif "model_state" in state:
                state = state["model_state"]
        missing, unexpected = fa_model.load_state_dict(state, strict=False)
        print(f"Loaded FA checkpoint: {args.fa_checkpoint}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: Checkpoint FA tidak ditemukan. Menggunakan bobot RANDOM.")

    fa_model.eval()

    # 2) Build jobs (single sample or dataset mode).
    jobs: List[Tuple[Path, str]] = []
    if args.prompts_csv:
        prompts_csv = Path(args.prompts_csv)
        if not prompts_csv.exists():
            raise FileNotFoundError(f"File CSV tidak ditemukan: {prompts_csv}")

        image_dir = Path(args.image_dir) if args.image_dir else None
        jobs = _collect_jobs_from_csv(
            prompts_csv=prompts_csv,
            image_col=args.image_col,
            prompt_col=args.prompt_col,
            image_dir=image_dir,
            max_samples=args.max_samples,
        )
    else:
        if not args.image_path or not args.prompt:
            raise ValueError("Mode single sample butuh --image-path dan --prompt.")

        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image tidak ditemukan: {image_path}")

        jobs = [(image_path, args.prompt)]

    print(f"Jumlah job: {len(jobs)}")

    # 3) Load SD pipeline ONCE for all jobs.
    first_config = ExtractionConfig(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        lora_weight_name=args.lora_weight_name,
        image_path=jobs[0][0],
        prompt=jobs[0][1],
        output_dir=output_dir,
        timestep=args.timestep,
        blocks=args.blocks,
        dtype=args.dtype,
        width=args.width,
        height=args.height,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Memulai pengunduhan/pemuatan model U-Net dari: {args.model_id}", flush=True)
    if torch.cuda.is_available():
        print(f"[{time.strftime('%H:%M:%S')}] VRAM sebelum load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    pipe, dtype = _load_pipeline(first_config)

    # --- TAMBAHKAN BLOK INI ---
    print("Memasang Hooks ke U-Net (Hanya Sekali)...", flush=True)
    feature_collector = BlockFeatureCollector(pipe.unet, blocks=args.blocks)
    attn_collector = AttentionCollector(pipe.unet)
    feature_collector.register()
    attn_collector.install()
    # --------------------------

    print(f"[{time.strftime('%H:%M:%S')}] Pemuatan U-Net SELESAI.", flush=True)
    if torch.cuda.is_available():
        print(f"[{time.strftime('%H:%M:%S')}] VRAM setelah load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # 4) Loop dataset/samples with shared pipeline.
    ok_count = 0
    fail_count = 0

    skip_count = 0

    try:
        for idx, (image_path, prompt) in enumerate(jobs, start=1):
            stem = f"{image_path.stem}_t{args.timestep}"
            z_path = output_dir / f"{stem}_fa_z{args.fa_z_dim}.npy"
            if z_path.exists() and z_path.stat().st_size > 0:
                skip_count += 1
                continue

            # print(f"\n[{idx}/{len(jobs)}] Processing: {image_path.name}", flush=True)

            config = ExtractionConfig(
                model_id=args.model_id,
                base_model_id=args.base_model_id,
                lora_weight_name=args.lora_weight_name,
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                timestep=args.timestep,
                blocks=args.blocks,
                dtype=args.dtype,
                width=args.width,
                height=args.height,
                device=args.device,
                local_files_only=args.local_files_only,
            )

            try:
                # Masukkan collector ke dalam argumen pemanggilan
                z_vector = run_extraction_with_pipeline(
                    pipe, dtype, config, fa_model, 
                    feature_collector, attn_collector # <--- TAMBAH INI
                )
                
                stem = f"{config.image_path.stem}_t{config.timestep}"
                z_path = output_dir / f"{stem}_fa_z{args.fa_z_dim}.npy"
                np.save(z_path, z_vector)
                # print(f"Tersimpan: {z_path}")
                # print(f"z shape : {z_vector.shape}")
                del z_vector
                ok_count += 1
                
                # --- SABUK PENGAMAN RESPAWNER ---
                if ok_count >= 30:
                    break
                # --------------------------------
            except Exception as exc:
                print(f"Gagal: {config.image_path.name} | {exc}")
                fail_count += 1
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSelesai.")
    print(f"Berhasil: {ok_count}")
    print(f"Dilewati: {skip_count}")
    print(f"Gagal   : {fail_count}")


if __name__ == "__main__":
    main()
