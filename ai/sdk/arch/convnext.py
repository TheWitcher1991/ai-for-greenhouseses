from sdk.contracts import BackboneConfig
from torchvision.models import convnext_tiny
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.feature_extraction import create_feature_extractor


def convnext_backbone(cfg: BackboneConfig):
    model = convnext_tiny(weights="DEFAULT" if cfg.pretrained else None)

    body = create_feature_extractor(
        model,
        return_nodes={
            "features.1": "0",
            "features.3": "1",
            "features.5": "2",
            "features.7": "3",
        },
    )

    return BackboneWithFPN(
        body,
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
    )
