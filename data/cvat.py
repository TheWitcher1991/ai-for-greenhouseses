import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from cvat_sdk import make_client

CVAT_URL = "cvat.stgau.ru"
USERNAME = "Ryabokonova.I"
PASSWORD = 'N$W"Ch|g9R'

OUTPUT_FILE = "annotations.json"
OUTPUT_IMAGES_DIR = Path("images")
OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)

client = make_client(host=CVAT_URL, credentials=(USERNAME, PASSWORD))
tasks = client.tasks.list()
TASK_IDS = [task.id for task in tasks]
print(TASK_IDS)

merged_coco = {"images": [], "annotations": [], "categories": []}
image_id_offset = 0
annotation_id_offset = 0
category_map = {}
category_id_counter = 1
image_file_counter = 0

for task_id in TASK_IDS:
    try:
        print(f"Работа с task {task_id}")
        task = client.tasks.retrieve(task_id)
        temp_zip_path = f"task_{task_id}.zip"
        task.export_dataset(format_name="COCO 1.0", filename=temp_zip_path)

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_ref.extractall(tmpdir)
                ann_path = Path(tmpdir) / "annotations" / "instances_default.json"

                try:
                    with open(ann_path, "r", encoding="utf-8") as f:
                        coco = json.load(f)
                except Exception as e:
                    print(f"Ошибка при чтении аннотаций для task {task_id}: {e}")
                    continue 

                for cat in coco.get("categories", []):
                    old_id = cat["id"]
                    if old_id not in category_map:
                        cat["id"] = category_id_counter
                        category_map[old_id] = category_id_counter
                        category_id_counter += 1
                        merged_coco["categories"].append(cat)

                for img in coco.get("images", []):
                    old_file_name = img["file_name"]
                    ext = Path(old_file_name).suffix
                    new_file_name = f"{image_file_counter:06d}{ext}"
                    image_file_counter += 1

                    src_img_path = Path(tmpdir) / "images" / "default" / old_file_name
                    dst_img_path = OUTPUT_IMAGES_DIR / new_file_name

                    try:
                        shutil.copy(src_img_path, dst_img_path)
                    except Exception as e:
                        print(f"Не удалось скопировать изображение {old_file_name}: {e}")
                        continue 

                    img["id"] += image_id_offset
                    img["file_name"] = new_file_name
                    merged_coco["images"].append(img)

                for ann in coco.get("annotations", []):
                    ann["id"] += annotation_id_offset
                    ann["image_id"] += image_id_offset
                    ann["category_id"] = category_map.get(ann["category_id"], ann["category_id"])
                    merged_coco["annotations"].append(ann)

                if merged_coco["images"]:
                    image_id_offset = max([img["id"] for img in merged_coco["images"]]) + 1
                if merged_coco["annotations"]:
                    annotation_id_offset = max([ann["id"] for ann in merged_coco["annotations"]]) + 1

    except Exception as e:
        print(f"Ошибка при обработке task {task_id}: {e}")
        continue 

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_coco, f, ensure_ascii=False, indent=2)

print(f"Объединённый COCO датасет сохранён в {OUTPUT_FILE}")
print(f"Изображения скопированы в папку {OUTPUT_IMAGES_DIR}")