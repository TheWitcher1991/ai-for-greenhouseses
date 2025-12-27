from framework.arch.convnext import convnext_backbone
from framework.arch.efficientnet import efficientnet_backbone
from framework.arch.resnet import resnet_backbone
from framework.contracts import BackboneAdapter, BackboneConfig
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
