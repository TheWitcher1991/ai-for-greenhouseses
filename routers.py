from fastapi import APIRouter, Depends, UploadFile
from starlette import status

from .dto import TrainDTO
from .services import TrainerService


class TrainerRouter:
    def __init__(self):
        self.router: APIRouter = APIRouter(
            prefix="/api/mlm/",
        )
        self.include_routers()

    def include_routers(self) -> None:
        self.train_route()
        self.predict_route()
        self.dataset_route()
        self.is_health_route()

    def train_route(self):
        @self.router.post("/train/")
        async def train(params: TrainDTO = Depends()):
            result = await TrainerService.train(**params.model_dump())
            return result

    def predict_route(self):
        @self.router.post("/predict/")
        async def predict(file: UploadFile):
            result = await TrainerService.predict(file)
            return result

    def dataset_route(self):
        @self.router.post("/dataset/")
        async def dataset(file: UploadFile):
            result = await TrainerService.dataset(file)
            return result

    def is_health_route(self):
        @self.router.get("/is-health/")
        async def is_health():
            return status.HTTP_200_OK
