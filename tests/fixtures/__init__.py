"""Reusable, outcome-safe test fixtures."""

from .datasets import conditional_annotation_fixture, lexical_fixture
from .encoders import FakeEncoder

__all__ = (
    "FakeEncoder",
    "conditional_annotation_fixture",
    "lexical_fixture",
)
