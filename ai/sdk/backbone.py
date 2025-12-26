from sdk.arch.convnext import convnext_backbone
from sdk.arch.efficientnet import efficientnet_backbone
from sdk.arch.resnet import resnet_backbone
from sdk.contracts import BackboneAdapter, BackboneConfig
from torch import nn


class BackboneBuilder(BackboneAdapter):

    @staticmethod
    def build(cfg: BackboneConfig) -> nn.Module:
        if cfg.name.startswith("resnet"):
            return resnet_backbone(cfg)

        if cfg.name.startswith("efficientnet"):
            return efficientnet_backbone(cfg)

        if cfg.name.startswith("convnext"):
            return convnext_backbone(cfg)

        raise ValueError(f"Unknown backbone: {cfg.name}")
