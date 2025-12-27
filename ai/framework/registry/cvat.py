import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import Task
from framework.contracts import RegistryAdapter, RegistryCredentials
from framework.logger import logger


class CvatRegistry(RegistryAdapter):

    def __init__(self, credentials: RegistryCredentials):
        self.host = credentials.get("host", "cvat.stgau.ru")
        self.output_file = credentials.get("output_annotations", "data/annotations.json")
        self.output_images = Path(credentials.get("output_images", "data/images"))
        self.output_images.mkdir(parents=True, exist_ok=True)

        self.merged_coco = {"images": [], "annotations": [], "categories": []}
        self.image_id_offset = 0
        self.annotation_id_offset = 0
        self.category_map = {}
        self.category_id_counter = 1
        self.image_file_counter = 0

        self.client = make_client(host=self.host, credentials=(credentials.get("login"), credentials.get("password")))

    def find_annotations(self) -> List[Task]:
        return self.client.tasks.list()

    def find_annotation(self, annotation_id: int) -> Task:
        return self.client.tasks.retrieve(annotation_id)

    def save_annotations(self) -> None:
        annotations = self.find_annotations()

        annotation_ids = [annotation.id for annotation in annotations]

        for annotation_id in annotation_ids:
            annotation = self.find_annotation(annotation_id)
            temp_zip_path = f"annotation_{annotation.id}.zip"
            annotation.export_dataset(format_name="COCO 1.0", filename=temp_zip_path)

            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_ref.extractall(tmpdir)
                    ann_path = Path(tmpdir) / "annotations" / "instances_default.json"

                    try:
                        with open(ann_path, "r", encoding="utf-8") as f:
                            coco = json.load(f)
                    except Exception as e:
                        logger.info(f"Ошибка при чтении аннотаций для task {annotation_id}: {e}")
                        continue

                    for cat in coco.get("categories", []):
                        old_id = cat["id"]
                        if old_id not in self.category_map:
                            cat["id"] = self.category_id_counter
                            self.category_map[old_id] = self.category_id_counter
                            self.category_id_counter += 1
                            self.merged_coco["categories"].append(cat)

                    for img in coco.get("images", []):
                        old_file_name = img["file_name"]
                        ext = Path(old_file_name).suffix
                        new_file_name = f"{self.image_file_counter:06d}{ext}"
                        self.image_file_counter += 1

                        src_img_path = Path(tmpdir) / "images" / "default" / old_file_name
                        dst_img_path = self.output_images / new_file_name

                        try:
                            shutil.copy(src_img_path, dst_img_path)
                        except Exception as e:
                            logger.info(f"Не удалось скопировать изображение {old_file_name}: {e}")
                            continue

                        img["id"] += self.image_id_offset
                        img["file_name"] = new_file_name
                        self.merged_coco["images"].append(img)

                    for ann in coco.get("annotations", []):
                        ann["id"] += self.annotation_id_offset
                        ann["image_id"] += self.image_id_offset
                        ann["category_id"] = self.category_map.get(ann["category_id"], ann["category_id"])
                        self.merged_coco["annotations"].append(ann)

                    if self.merged_coco["images"]:
                        self.image_id_offset = max(img["id"] for img in self.merged_coco["images"]) + 1

                    if self.merged_coco["annotations"]:
                        self.annotation_id_offset = max(ann["id"] for ann in self.merged_coco["annotations"]) + 1

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.merged_coco, f, ensure_ascii=False, indent=2)

        logger.info(f"Объединённый COCO датасет сохранён в {self.output_file}")
        logger.info(f"Изображения скопированы в папку {self.output_images}")
