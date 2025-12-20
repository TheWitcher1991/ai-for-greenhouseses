from dataset import CocoSegmentationDataset

ds = CocoSegmentationDataset("data/images", "data/cvat_merged_coco.json")

for i in range(len(ds)):
    try:
        ds[i]
    except Exception as e:
        print(f"Ошибка на {i}: {e}")
