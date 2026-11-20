# -*- coding: utf-8 -*-
from .e2former import E2former
from .e2former_wrapper import E2FormerBackbone
from .e2former_mixed_resolution import E2formerMR

E2FormerV2 = E2former
E2FormerV2Backbone = E2FormerBackbone

__all__ = [
    "E2former",
    "E2formerMR",
    "E2FormerBackbone",
    "E2FormerV2",
    "E2FormerV2Backbone",
]
