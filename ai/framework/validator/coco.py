import json
import os

from framework.contracts import DatasetValidatorAdapter


class CocoDatasetValidator(DatasetValidatorAdapter):

    def validate(self, dataset_path: str, images_path: str):
        with open(dataset_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        errors = []

        category_ids = [cat["id"] for cat in coco.get("categories", [])]
        if len(category_ids) != len(set(category_ids)):
            errors.append("Дубликаты category id")

        image_map = {img["id"]: img for img in coco.get("images", [])}

        for img in coco["images"]:
            path = os.path.join(images_path, img["file_name"])
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

        return errors
