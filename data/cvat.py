import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from cvat_sdk import make_client

CVAT_URL = "cvat.stgau.ru"

# Данные для авторизации
USERNAME = "Ryabokonova.I"
PASSWORD = 'N$W"Ch|g9R'

OUTPUT_FILE = "cvat_merged_coco.json"
OUTPUT_IMAGES_DIR = Path("images")

OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)

client = make_client(host=CVAT_URL, credentials=(USERNAME, PASSWORD))

tasks = client.tasks.list()
TASK_IDS = [
    96,
    95,
    94,
    84,
    83,
    82,
    81,
    80,
    79,
    78,
    77,
    76,
    75,
    74,
    73,
    72,
    71,
    70,
    69,
    68,
    67,
    66,
    65,
    64,
    63,
    62,
    61,
    60,
    59,
    58,
    57,
    56,
    55,
    54,
    53,
    52,
    51,
    50,
    49,
    48,
    47,
    46,
    45,
    44,
    43,
    42,
    41,
    40,
    39,
    38,
    37,
    36,
    35,
    34,
    33,
    32,
    31,
    30,
    29,
    28,
    11,
]

print(TASK_IDS)

merged_coco = {"images": [], "annotations": [], "categories": []}

image_id_offset = 0
annotation_id_offset = 0
category_map = {}
category_id_counter = 1
image_file_counter = 0

for task_id in TASK_IDS:
    print(f"Работа с task {task_id}")
    task = client.tasks.retrieve(task_id)
    temp_zip_path = f"task_{task_id}.zip"
    task.export_dataset(format_name="COCO 1.0", filename=temp_zip_path)

    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_ref.extractall(tmpdir)
            ann_path = Path(tmpdir) / "annotations" / "instances_default.json"

            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)

            for cat in coco["categories"]:
                old_id = cat["id"]
                if old_id not in category_map:
                    cat["id"] = category_id_counter
                    category_map[old_id] = category_id_counter
                    category_id_counter += 1
                    merged_coco["categories"].append(cat)

            for img in coco["images"]:
                old_file_name = img["file_name"]
                new_file_name = f"{image_file_counter:06d}_{old_file_name}"
                image_file_counter += 1

                src_img_path = Path(tmpdir) / "images" / "default" / old_file_name
                dst_img_path = OUTPUT_IMAGES_DIR / new_file_name
                shutil.copy(src_img_path, dst_img_path)

                img["id"] += image_id_offset
                img["file_name"] = new_file_name
                merged_coco["images"].append(img)

            for ann in coco["annotations"]:
                ann["id"] += annotation_id_offset
                ann["image_id"] += image_id_offset
                ann["category_id"] = category_map[ann["category_id"]]
                merged_coco["annotations"].append(ann)

            image_id_offset = max([img["id"] for img in merged_coco["images"]]) + 1
            annotation_id_offset = max([ann["id"] for ann in merged_coco["annotations"]]) + 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_coco, f, ensure_ascii=False, indent=2)

print(f"Объединённый COCO датасет сохранён в {OUTPUT_FILE}")
print(f"Изображения скопированы в папку {OUTPUT_IMAGES_DIR}")
