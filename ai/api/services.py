import shutil
from threading import Thread
from typing import Dict

from fastapi import UploadFile
from sdk.logger import logger
from sdk.transforms import ComposeTransforms
from sdk.models.v1.dataset.coco import CocoSegmentationDataset
from sdk.models.v1.ml import MLM


class TrainerService:

    @staticmethod
    async def train(params: Dict):
        async def _run():
            try:
                dataset = CocoSegmentationDataset(
                    images_dir="data/v1/images",
                    annotation_file="data/v1/annotations.json",
                    transforms=ComposeTransforms(),
                )

                trainer = MLM(
                    dataset=dataset,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                )

                trainer.train()
                trainer.save()
            except Exception as e:
                logger.error(str(e))

        Thread(target=_run, daemon=True).start()

        return {"status": "Обучение начато"}

    @staticmethod
    async def predict(file: UploadFile):
        try:
            mlm = MLM()
            mlm.load()
            return mlm.predict("data/test.jpg")
        except Exception as e:
            logger.error(str(e))

    @staticmethod
    async def dataset(file: UploadFile):
        DATASET_DIR = "uploads/"
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        path = DATASET_DIR / file.filename

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"status": "Датасет загружен"}
