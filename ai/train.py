from sdk.contracts import BackboneConfig, BackboneType
from sdk.models.v1.dataset.coco import CocoSegmentationDataset
from sdk.models.v1.ml import MLM
from sdk.transforms import ComposeTransforms

dataset = CocoSegmentationDataset(
    images_dir="data/v1/images", annotation_file="data/v1/annotations.json", transforms=ComposeTransforms()
)

backbone_cfg = BackboneConfig(name=BackboneType.resnet50, pretrained=True)

trainer = MLM(dataset=dataset, backbone_cfg=backbone_cfg)

trainer.train()

trainer.save()
