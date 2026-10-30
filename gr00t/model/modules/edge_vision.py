"""EE-VLA edge vision encoders for GR00T N1.7.

Only the VLM views (the first `edge_vlm_views` video keys) go through Cosmos-Reason2; every other
("edge") view gets its own trainable DINOv3 ViT-B/16. Each edge view gives one token: CLS ->
MLP (768 -> 2048 -> GELU -> 2048) -> LayerNorm. The tokens are appended to the VL tokens after
the action head's vlln + VL self-attention, as image tokens of the DiT cross-attention memory.
Training-time edge dropout: with probability `dropout_p` a sample has all its edge tokens
removed from the cross-attention mask ("mask" mode, per sample).
"""

import timm
import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


class EdgeVisionEncoder(nn.Module):
    def __init__(
        self,
        num_views: int,
        out_dim: int,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        dropout_p: float = 0.2,
        dropout_unit: str = "sample",
        tune: bool = True,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        assert dropout_unit in ("sample", "view"), dropout_unit
        self.num_views, self.model_name = num_views, model_name
        self.dropout_p, self.dropout_unit = dropout_p, dropout_unit
        # Built without weights: load_pretrained_encoders() loads DINOv3 when the checkpoint
        # being loaded has no edge encoders.
        self.encoders = nn.ModuleList(
            [timm.create_model(model_name, pretrained=False, num_classes=0) for _ in range(num_views)]
        )
        enc = self.encoders[0]
        dim = enc.num_features
        cfg = enc.pretrained_cfg
        self.image_size = tuple(cfg.get("input_size", (3, 256, 256))[1:])
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim))
             for _ in range(num_views)]
        )
        if not tune:
            self.encoders.requires_grad_(False)
        elif grad_checkpointing:
            for e in self.encoders:
                e.set_grad_checkpointing(True)

    def load_pretrained_encoders(self):
        """Load the pretrained timm weights into every encoder (in place)."""
        state = timm.create_model(self.model_name, pretrained=True, num_classes=0).state_dict()
        for enc in self.encoders:
            enc.load_state_dict(state)

    def reset_projections(self):
        for proj in self.proj:
            for m in proj:
                if isinstance(m, (nn.Linear, nn.LayerNorm)):
                    m.reset_parameters()

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, V, 3, H, W] uint8 -> edge tokens [B, V, out_dim]."""
        b, v = pixels.shape[:2]
        assert v == self.num_views, f"expected {self.num_views} edge views, got {v}"
        dtype = next(self.encoders[0].parameters()).dtype
        x = pixels.flatten(0, 1).to(dtype=torch.float32) / 255.0
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False, antialias=True)
        x = ((x - self.mean) / self.std).unflatten(0, (b, v))
        tokens = [self.proj[i](self.encoders[i].forward_features(x[:, i].to(dtype))[:, :1]) for i in range(v)]
        return torch.cat(tokens, dim=1)

    def tokens_and_keep(self, pixels: torch.Tensor, dtype: torch.dtype):
        """Edge tokens [B, V, D] and their keep mask [B, V] (training-time dropout applied)."""
        tokens = self.encode(pixels).to(dtype)
        b, v = tokens.shape[:2]
        keep = torch.ones(b, v, dtype=torch.bool, device=tokens.device)
        if self.training and self.dropout_p > 0:
            shape = (b, 1) if self.dropout_unit == "sample" else (b, v)
            keep = (torch.rand(shape, device=tokens.device) >= self.dropout_p).expand(b, v)
        return tokens, keep

    def route(self, backbone_output: BatchFeature, pixels: torch.Tensor) -> BatchFeature:
        """memory = [VL tokens | edge tokens]; image_mask += 1; attention_mask += keep."""
        tokens, keep = self.tokens_and_keep(pixels, backbone_output["backbone_features"].dtype)
        b, v = tokens.shape[:2]
        backbone_output["backbone_features"] = torch.cat([backbone_output["backbone_features"], tokens], dim=1)
        backbone_output["backbone_attention_mask"] = torch.cat(
            [backbone_output["backbone_attention_mask"].bool(), keep], dim=1)
        backbone_output["image_mask"] = torch.cat(
            [backbone_output["image_mask"], torch.ones(b, v, dtype=torch.bool, device=tokens.device)], dim=1)
        return backbone_output
