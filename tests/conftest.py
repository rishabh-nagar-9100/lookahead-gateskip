"""Shared fixtures: tiny random Llama and Qwen2 models (CPU, fp32, nothing downloaded)."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest

from lagate.models.tiny import build_tiny


@pytest.fixture(params=["llama", "qwen2"])
def tiny_model(request):
    return build_tiny(request.param)
