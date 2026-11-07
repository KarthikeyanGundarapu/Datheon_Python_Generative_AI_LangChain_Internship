"""Validated output schemas for invoice and resume extraction."""
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Invoice(StrictModel):
    vendor: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = Field(None, description="ISO 8601 date when available")
    due_date: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    line_items: list[dict[str, str | float | None]] = Field(default_factory=list)


class Resume(StrictModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[dict[str, str | None]] = Field(default_factory=list)
    education: list[dict[str, str | None]] = Field(default_factory=list)
