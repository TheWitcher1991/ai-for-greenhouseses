from dataset import CocoSegmentationDataset

ds = CocoSegmentationDataset("data/images", "data/annotations.json")

for i in range(len(ds)):
    try:
        print(ds[i])
    except Exception as e:
        print(f"Ошибка на {i}: {e}")
