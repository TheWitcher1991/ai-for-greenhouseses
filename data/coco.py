import json
import os

PATH_TO_JSON = "cvat_merged_coco.json"
IMAGES_DIR = "images/"

with open(PATH_TO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

errors = []

category_ids = [cat["id"] for cat in coco.get("categories", [])]
if len(category_ids) != len(set(category_ids)):
    errors.append("Дубликаты category id")

image_map = {img["id"]: img for img in coco.get("images", [])}

for img in coco["images"]:
    path = os.path.join(IMAGES_DIR, img["file_name"])
    if not os.path.exists(path):
        errors.append(f"Файл отсутствует: {path}")

for ann in coco["annotations"]:
    image_id = ann["image_id"]

    if image_id not in image_map:
        errors.append(f"Аннотация на image_id={image_id}, которого нет")

    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        errors.append(f"Неверный bbox: {bbox}")
        continue

    x, y, w, h = bbox

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        errors.append(f"Отрицательный или нулевой bbox: {bbox}")

    img = image_map[image_id]
    if x + w > img["width"] or y + h > img["height"]:
        errors.append(f"bbox выходит за границы: {bbox} > image {img['file_name']}")

print("\nРЕЗУЛЬТАТ")
if not errors:
    print("Ошибок не найдено")
else:
    print(f"Найдено ошибок: {len(errors)}")
    for e in errors:
        print("•", e)
