import math
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TensorDict = Dict[str, torch.Tensor]

# ============================================================
# UTILS & PREPROCESSING
# ============================================================
def _first_tensor(value: object) -> Optional[torch.Tensor]:
    """Return first tensor found inside nested outputs."""
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

def _preprocess_image(image: Image.Image, width: int = 512, height: int = 512) -> torch.Tensor:
    image = image.convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, H, W]
    image_tensor = torch.from_numpy(arr)
    image_tensor = image_tensor * 2.0 - 1.0
    return image_tensor

def _encode_latent(pipe, image_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    image_tensor = image_tensor.to(pipe.device, dtype=dtype)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(image_tensor).latent_dist
        latents = latent_dist.sample() * pipe.vae.config.scaling_factor
    return latents

def _build_text_embeddings(pipe, prompt: str) -> torch.Tensor:
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

def _add_noise_at_timestep(pipe, latents: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(latents)
    t = torch.full((latents.shape[0],), timestep, dtype=torch.long, device=latents.device)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
    return noisy_latents, t

# ============================================================
# EXTRACTORS (U-Net Hooks)
# ============================================================
class BlockFeatureCollector:
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
    def __init__(self, name: str, store: TensorDict):
        super().__init__()
        self.name = name
        self.store = store

    def _save_spatial_attention(self, attention_probs: torch.Tensor, batch_size: int, spatial_h: Optional[int], spatial_w: Optional[int]) -> None:
        heads = attention_probs.shape[0] // batch_size
        q_tokens = attention_probs.shape[1]
        k_tokens = attention_probs.shape[2]
        probs = attention_probs.view(batch_size, heads, q_tokens, k_tokens)
        saliency = probs.mean(dim=1).mean(dim=-1)
        if spatial_h is None or spatial_w is None:
            side = int(math.sqrt(q_tokens))
            if side * side == q_tokens:
                spatial_h, spatial_w = side, side
            else:
                spatial_h, spatial_w = 1, q_tokens
        saliency_map = saliency.view(batch_size, 1, spatial_h, spatial_w)
        self.store[self.name] = saliency_map.detach()

    def __call__(self, attn: nn.Module, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross_attention = encoder_hidden_states is not None

        spatial_h, spatial_w = None, None
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

# ============================================================
# FEATURE AGGREGATION ENCODER (FA)
# ============================================================
class SpatialAttention(nn.Module):
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
    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 4, spatial_max_hw: int = 32):
        super().__init__()
        self.spatial = SpatialAttention(in_channels, num_heads=num_heads, max_hw=spatial_max_hw)
        self.channel = ChannelAttention(in_channels, reduction=reduction)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial(x) + self.channel(x)
        return self.norm(out)

class DFATB(nn.Module):
    def __init__(self, channel_list: Sequence[int], num_heads: int = 4, reduction: int = 4, spatial_max_hw: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            DFATBLayer(c, num_heads=num_heads, reduction=reduction, spatial_max_hw=spatial_max_hw)
            for c in channel_list
        ])

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(fm) for layer, fm in zip(self.layers, feature_maps)]

class FAFNLayer(nn.Module):
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
    def __init__(self, channel_list: Sequence[int], expansion: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([FAFNLayer(c, expansion=expansion) for c in channel_list])

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(fm) for layer, fm in zip(self.layers, feature_maps)]

class DifferentialDenoising(nn.Module):
    def __init__(self, feature_channels: Sequence[int], attn_channels: Sequence[int], lambda_init: float = 0.5):
        super().__init__()
        self.lambdas = nn.ParameterList(
            [nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32)) for _ in feature_channels]
        )
        self.attn_proj = nn.ModuleList(
            [nn.Conv2d(ac, fc, 1, bias=False) for fc, ac in zip(feature_channels, attn_channels)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm2d(fc) for fc in feature_channels])

    def forward(self, ffafn_list: List[torch.Tensor], attn_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, fdenoise_list: List[torch.Tensor], adiff_list: List[torch.Tensor]) -> torch.Tensor:
        vj = [self.gap(f).flatten(1) for f in fdenoise_list]
        ul = [self.gap(a).flatten(1) for a in adiff_list]
        return torch.cat(vj + ul, dim=1)

class BottleneckProjection(nn.Module):
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
    def __init__(self, feature_channels: Sequence[int], attn_channels: Sequence[int], z_dim: int = 128, num_heads: int = 4, fafn_expansion: int = 2, lambda_init: float = 0.5, dropout: float = 0.1, spatial_max_hw: int = 32):
        super().__init__()
        self.dfatb = DFATB(feature_channels, num_heads=num_heads, spatial_max_hw=spatial_max_hw)
        self.fafn = FAFN(feature_channels, expansion=fafn_expansion)
        self.diff_denoise = DifferentialDenoising(feature_channels, attn_channels, lambda_init=lambda_init)
        self.gap_concat = GAPConcat()

        c_tot = sum(feature_channels) + sum(attn_channels)
        self.bottleneck = BottleneckProjection(c_tot, out_dim=z_dim, dropout=dropout)

    def forward(self, feature_maps: List[torch.Tensor], attn_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        f_star = self.dfatb(feature_maps)
        f_fafn = self.fafn(f_star)
        f_denoise, a_diff = self.diff_denoise(f_fafn, attn_pairs)
        z_fused = self.gap_concat(f_denoise, a_diff)
        z = self.bottleneck(z_fused)
        return {"z": z, "z_fused": z_fused}

# ============================================================
# MLP HEAD
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = 6):
        super().__init__()
        MLP_HIDDEN = 512
        DROPOUT = 0.3
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
# ALIGNMENT UTILITIES
# ============================================================
def _prepare_sorted_maps(feature_maps: TensorDict, attention_maps: TensorDict) -> Tuple[List[str], List[torch.Tensor], List[str], List[torch.Tensor]]:
    feature_items = [(k, v.float()) for k, v in sorted(feature_maps.items()) if isinstance(v, torch.Tensor) and v.dim() == 4]
    if not feature_items:
        raise RuntimeError("Tidak ada feature maps 4D.")
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

def _build_attention_pairs(feature_list: List[torch.Tensor], attention_keys: List[str], attention_list: List[torch.Tensor]) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[int], List[Tuple[str, str]]]:
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    attn_channels: List[int] = []
    pair_names: List[Tuple[str, str]] = []

    attn_areas = [a.shape[-2] * a.shape[-1] for a in attention_list]
    for f in feature_list:
        target_area = f.shape[-2] * f.shape[-1]
        ranked_idx = sorted(range(len(attention_list)), key=lambda i: abs(attn_areas[i] - target_area))
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

def _align_features_to_model(feature_keys: List[str], feature_list: List[torch.Tensor], expected_channels: List[int]) -> Tuple[List[str], List[torch.Tensor]]:
    if len(feature_list) < len(expected_channels):
        raise RuntimeError(f"Jumlah feature map ({len(feature_list)}) lebih kecil dari kebutuhan model ({len(expected_channels)}).")
    
    selected_keys: List[str] = []
    selected_maps: List[torch.Tensor] = []
    used_indices: set[int] = set()

    for target_c in expected_channels:
        best_idx = None
        best_score = None
        for idx, fmap in enumerate(feature_list):
            if idx in used_indices: continue
            score = (abs(fmap.shape[1] - target_c), -(fmap.shape[-2] * fmap.shape[-1]))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            raise RuntimeError("Gagal memilih feature map.")
        used_indices.add(best_idx)
        selected_keys.append(feature_keys[best_idx])
        selected_maps.append(feature_list[best_idx])
    return selected_keys, selected_maps

def _adapt_attention_channels(attn: torch.Tensor, target_channels: int) -> torch.Tensor:
    current_channels = attn.shape[1]
    if current_channels == target_channels:
        return attn
    if target_channels <= 0:
        raise ValueError("target_channels > 0 required.")
    if current_channels == 1 and target_channels > 1:
        return attn.expand(-1, target_channels, -1, -1)
    if current_channels > target_channels:
        return attn[:, :target_channels, :, :]

    repeat_factor = (target_channels + current_channels - 1) // current_channels
    expanded = attn.repeat(1, repeat_factor, 1, 1)
    return expanded[:, :target_channels, :, :]
