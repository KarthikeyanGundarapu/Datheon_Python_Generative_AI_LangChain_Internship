"""Shared OpenAI client helpers."""
import os
from openai import OpenAI


def client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file.")
    return OpenAI()


def model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
