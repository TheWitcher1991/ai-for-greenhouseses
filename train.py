from sdk.dataset import CocoSegmentationDataset
from sdk.ml import MLM
from sdk.transforms import ComposeTransforms

dataset = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

model = MLM(dataset=dataset)

model.train()

model.save()
