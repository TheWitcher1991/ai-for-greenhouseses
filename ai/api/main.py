from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .routers import TrainerRouter

app = FastAPI(
    openapi_version="3.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc/",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(TrainerRouter().router)
