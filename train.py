from sdk.dataset import CocoSegmentationDataset
from sdk.ml import MLModel
from sdk.transforms import ComposeTransforms

dataset = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

ml = MLModel(dataset=dataset)

ml.train()

ml.save()
