from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


CameraName = Literal["top", "wrist", "overhead", "left_wrist", "right_wrist"]


class Box(BaseModel):
    """Pixel-space bounding box in [x1, y1, x2, y2] format."""

    values: list[int] = Field(min_length=4, max_length=4)

    @field_validator("values")
    @classmethod
    def _validate_box(cls, values: list[int]) -> list[int]:
        x1, y1, x2, y2 = [int(v) for v in values]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        if x1 < 0 or y1 < 0:
            raise ValueError("box coordinates must be non-negative")
        return [x1, y1, x2, y2]


class CameraObservation(BaseModel):
    observation_id: str
    camera_id: str
    image_path: str
    width: int
    height: int
    timestamp: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoxLayer(BaseModel):
    box_layer_id: str
    camera_id: str
    reference_observation_id: str
    reference_image_path: str
    preview_overlay_id: str
    preview_overlay_path: str
    red_box: list[int] = Field(min_length=4, max_length=4)
    green_box: list[int] = Field(min_length=4, max_length=4)
    width: int
    height: int
    red_label: str = "source"
    green_label: str = "target"
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("red_box", "green_box")
    @classmethod
    def _validate_layer_box(cls, values: list[int]) -> list[int]:
        return Box(values=values).values


class BoxOverlay(BaseModel):
    box_overlay_id: str
    box_layer_id: str | None = None
    camera_id: str
    observation_id: str
    original_image_path: str
    overlay_path: str
    red_box: list[int] = Field(min_length=4, max_length=4)
    green_box: list[int] = Field(min_length=4, max_length=4)
    width: int
    height: int
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("red_box", "green_box")
    @classmethod
    def _validate_overlay_box(cls, values: list[int]) -> list[int]:
        return Box(values=values).values


class ExecutionResult(BaseModel):
    success: bool
    instruction: str
    box_overlay_id: str
    box_layer_id: str | None = None
    atomic_action: str | None = None
    backend: str
    result_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
