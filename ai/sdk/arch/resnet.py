from sdk.contracts import BackboneConfig, BackboneType
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def resnet_backbone(cfg: BackboneConfig):
    if cfg.name == BackboneType.resnet50:
        model = resnet50(weights="DEFAULT" if cfg.pretrained else None)
        out_channels = 256
    elif cfg.name == BackboneType.resnet101:
        model = resnet101(weights="DEFAULT" if cfg.pretrained else None)
    elif cfg.name == BackboneType.resnet101:
        model = resnet152(weights="DEFAULT" if cfg.pretrained else None)
        out_channels = 256
    else:
        raise ValueError(cfg.name)

    return BackboneWithFPN(
        model,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=out_channels,
    )
