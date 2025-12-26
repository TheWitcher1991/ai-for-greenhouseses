from sdk.transforms import ComposeTransforms
from sdk.v2.dataset.coco import CocoSegmentationDataset

ds = CocoSegmentationDataset(
    images_dir="data/v2/images", annotation_file="data/v2/annotations.json", transforms=ComposeTransforms()
)

for i in range(len(ds)):
    try:
        ds[i]
    except Exception as e:
        print(f"Ошибка на {i}: {e}")
