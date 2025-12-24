from sdk.transforms import ComposeTransforms
from sdk.v1.dataset import CocoSegmentationDataset
from sdk.v1.ml import MLM

dataset = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

trainer = MLM(
    dataset=dataset,
)

trainer.train()
trainer.save()
