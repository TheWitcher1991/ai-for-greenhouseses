from sdk.transforms import ComposeTransforms
from sdk.v1.dataset.coco import CocoSegmentationDataset
from sdk.v1.ml import MLM

dataset = CocoSegmentationDataset(
    images_dir="data/v1/images", annotation_file="data/v1/annotations.json", transforms=ComposeTransforms()
)

trainer = MLM(
    dataset=dataset,
)

trainer.train()
trainer.save()
