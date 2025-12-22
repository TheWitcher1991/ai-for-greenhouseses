from sdk.transforms import ComposeTransforms
from sdk.v1.dataset import CocoSegmentationDataset
from sdk.v1.ml import MLM

dataset = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

model = MLM(dataset=dataset)

model.train()

model.save()
