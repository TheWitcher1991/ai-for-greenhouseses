from framework.transforms import ComposeTransforms
from framework.detection.v1.dataset.coco import CocoSegmentationDataset

ds = CocoSegmentationDataset(
    images_dir="data/v1/images", annotation_file="data/v1/annotations.json", transforms=ComposeTransforms()
)

for i in range(len(ds)):
    try:
        ds[i]
    except Exception as e:
        print(f"Ошибка на {i}: {e}")
