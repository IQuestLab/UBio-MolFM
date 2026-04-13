# -*- coding: utf-8 -*-
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

# Pydantic emits noisy schema warnings on import under Python 3.12.
# They do not affect runtime behavior here, so keep stdout/stderr clean.
warnings.filterwarnings(
    "ignore",
    category=UnsupportedFieldAttributeWarning,
)

from .models import E2FormerBackbone, E2FormerV2, E2FormerV2Backbone, E2former

__all__ = [
    "E2former",
    "E2FormerBackbone",
    "E2FormerV2",
    "E2FormerV2Backbone",
]
