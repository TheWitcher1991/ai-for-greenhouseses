from framework.contracts import BackboneConfig, BackboneType
from framework.detection.v1.ml import MLM

backbone_cfg = BackboneConfig(name=BackboneType.resnet50, pretrained=True)

predictor = MLM(backbone_cfg=backbone_cfg)

predictor.load()

predictor.predict("data/test.jpg")
