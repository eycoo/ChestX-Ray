import gc
import torch
import numpy as np
from PIL import Image

from diffusers import StableDiffusionPipeline
from models import (
    FeatureAggregationEncoder,
    MLPClassifier,
    BlockFeatureCollector,
    AttentionCollector,
    _preprocess_image,
    _encode_latent,
    _build_text_embeddings,
    _add_noise_at_timestep,
    _prepare_sorted_maps,
    _build_attention_pairs,
    _model_feature_channels,
    _model_attention_channels,
    _align_features_to_model,
    _adapt_attention_channels
)

class MedicalXRayPipeline:
    def __init__(self, sd_model_id: str, lora_weight: str, pt_path: str, mlp_path: str = None, device: str = "cuda"):
        self.device = device
        self.sd_model_id = sd_model_id
        self.lora_weight = lora_weight
        self.pt_path = pt_path
        self.mlp_path = mlp_path
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        self.class_names = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']
        
        self.pipe = None
        self.fa_model = None
        self.cls_head = None
        
        self.feature_collector = None
        self.attn_collector = None

    def load_models(self):
        print("Loading Stable Diffusion Pipeline...")
        # Load SD U-Net & VAE
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=self.dtype
        ).to(self.device)
        
        try:
            self.pipe.load_lora_weights(self.sd_model_id, weight_name=self.lora_weight)
        except Exception as e:
            print(f"Failed to load LoRA weights. Proceeding without LoRA: {e}")

        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        for p in self.pipe.unet.parameters():
            p.requires_grad = False
        for p in self.pipe.vae.parameters():
            p.requires_grad = False
        for p in self.pipe.text_encoder.parameters():
            p.requires_grad = False

        print("Attaching Hooks...")
        self.feature_collector = BlockFeatureCollector(self.pipe.unet, blocks="up_mid")
        self.attn_collector = AttentionCollector(self.pipe.unet)
        self.feature_collector.register()
        self.attn_collector.install()

        print("Loading FA Model & Classifier Component...")
        ckpt = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        
        self.fa_model = FeatureAggregationEncoder(
            feature_channels=ckpt.get("feature_channels", [320, 640, 1280, 1280]),
            attn_channels=ckpt.get("attn_channels", [320, 640, 1280, 1280]),
            z_dim=ckpt.get("fa_z_dim", 128),
            num_heads=4,
            fafn_expansion=2,
            lambda_init=0.5,
            dropout=0.1,
            spatial_max_hw=32
        ).to(self.device)
        self.fa_model.load_state_dict(ckpt["fa_model_state"])
        self.fa_model.eval()
        for p in self.fa_model.parameters():
            p.requires_grad = False

        self.cls_head = MLPClassifier(
            feat_dim=ckpt.get("fa_z_dim", 128) if isinstance(ckpt, dict) and "fa_z_dim" in ckpt else 128,
            num_classes=ckpt.get("num_classes", 6) if isinstance(ckpt, dict) and "num_classes" in ckpt else 6
        ).to(self.device).eval()

        if self.mlp_path and os.path.exists(self.mlp_path):
            mlp_ckpt = torch.load(self.mlp_path, map_location=self.device)
            if "state_dict" in mlp_ckpt:
                self.cls_head.load_state_dict(mlp_ckpt["state_dict"])
            elif "cls_head_state" in mlp_ckpt:
                self.cls_head.load_state_dict(mlp_ckpt["cls_head_state"])
            else:
                self.cls_head.load_state_dict(mlp_ckpt)
        else:
            # Fallback
            if isinstance(ckpt, dict) and "cls_head_state" in ckpt:
                try:
                    self.cls_head.load_state_dict(ckpt["cls_head_state"])
                except Exception as e:
                    print(f"Warning: Discarding incompatible cls_head_state from FA checkpoint. ({e})")
        
        for p in self.cls_head.parameters():
            p.requires_grad = False

        if "class_names" in ckpt:
            self.class_names = ckpt["class_names"]

        print("Models loaded successfully.")

    def cleanup(self):
        self.feature_collector.features.clear()
        self.attn_collector.maps.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image: Image.Image, prompt: str = "A chest X-ray", timestep: int = 10):
        try:
            image_tensor = _preprocess_image(image, 512, 512)
            latents = _encode_latent(self.pipe, image_tensor=image_tensor, dtype=self.dtype)
            text_embeddings = _build_text_embeddings(self.pipe, prompt=prompt)
            noisy_latents, t = _add_noise_at_timestep(self.pipe, latents=latents, timestep=timestep)

            # U-Net Forward Pass (Populates hooks)
            with torch.no_grad():
                _ = self.pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings, return_dict=True)

            feature_maps = {k: v for k, v in self.feature_collector.features.items() if v.dim() == 4}
            attention_maps = {k: v for k, v in self.attn_collector.maps.items() if v.dim() == 4}

            if not feature_maps:
                raise RuntimeError("No 4D feature maps captured.")

            feature_keys, feature_list, attention_keys, attention_list = _prepare_sorted_maps(feature_maps, attention_maps)
            expected_feature_channels = _model_feature_channels(self.fa_model)
            selected_feature_keys, selected_feature_list = _align_features_to_model(
                feature_keys, feature_list, expected_feature_channels
            )

            attn_pairs, _, _ = _build_attention_pairs(
                selected_feature_list, attention_keys, attention_list
            )
            
            expected_attn_channels = _model_attention_channels(self.fa_model)
            aligned_pairs = []
            for (a1, a2), target_attn_c in zip(attn_pairs, expected_attn_channels):
                aligned_pairs.append((
                    _adapt_attention_channels(a1, target_attn_c),
                    _adapt_attention_channels(a2, target_attn_c)
                ))

            with torch.no_grad():
                f_dev = [f.to(self.device) for f in selected_feature_list]
                a_dev = [(a1.to(self.device), a2.to(self.device)) for (a1, a2) in aligned_pairs]

                # Enable AMP if on CUDA
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', dtype=torch.float16, enabled=self.device == "cuda"):
                    fa_outputs = self.fa_model(f_dev, a_dev)
                    logits = self.cls_head(fa_outputs["z"])
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    confidence = float(np.max(probs))
                    class_idx = int(np.argmax(probs))
                    class_name = self.class_names[class_idx]

            return {
                "prediction": class_name,
                "confidence": confidence,
                "probabilities": {name: float(prob) for name, prob in zip(self.class_names, probs)}
            }
        finally:
            self.cleanup()
