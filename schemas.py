from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GenerationMode = Literal["txt2img", "img2img"]


class GenerateRequest(BaseModel):
    mode: GenerationMode = "txt2img"
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    image: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    strength: float | None = None
    model_id: str | None = None
    lora_path: str | None = None
    lora_scale: float | None = None


class GenerateResponse(BaseModel):
    status: Literal["ok"]
    worker_id: str
    mode: GenerationMode
    model_id: str
    output_path: str
    metadata_path: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    worker_id: str


class ActiveTaskResponse(BaseModel):
    status: str
    stage: str
    detail: str | None = None
    model_id: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    last_output_path: str | None = None


class ServerInfoResponse(BaseModel):
    backend: str
    version: str
    worker_id: str
    configured_model_id: str
    loaded_model_id: str | None = None
    device: str
    dtype: str
    output_dir: str
    supported_modes: list[str]
    extra: dict[str, Any] = Field(default_factory=dict)
