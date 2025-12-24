from sdk.transforms import ComposeTransforms
from sdk.v1.dataset import CocoSegmentationDataset

ds = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

for i in range(len(ds)):
    try:
        ds[i]
    except Exception as e:
        print(f"Ошибка на {i}: {e}")
