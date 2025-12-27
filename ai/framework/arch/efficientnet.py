from framework.contracts import BackboneConfig, BackboneType
from torchvision.models import efficientnet_b3, efficientnet_b4
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.feature_extraction import create_feature_extractor


def efficientnet_backbone(cfg: BackboneConfig):
    if cfg.name == BackboneType.efficientnet_b3:
        model = efficientnet_b3(weights="DEFAULT" if cfg.pretrained else None)
        in_channels_list = [48, 136, 384]
    elif cfg.name == BackboneType.efficientnet_b4:
        model = efficientnet_b4(weights="DEFAULT" if cfg.pretrained else None)
        in_channels_list = [56, 160, 448]
    else:
        raise ValueError(cfg.name)

    body = create_feature_extractor(
        model,
        return_nodes={
            "features.2": "0",
            "features.4": "1",
            "features.6": "2",
        },
    )

    backbone = BackboneWithFPN(
        body,
        # return_layers={"0": 0, "1": 1, "2": 2},
        in_channels_list=in_channels_list,
        out_channels=256,
    )

    return backbone
