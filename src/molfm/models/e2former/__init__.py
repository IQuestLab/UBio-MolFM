# -*- coding: utf-8 -*-
from .e2former import E2former
from .e2former_wrapper import E2FormerBackbone

E2FormerV2 = E2former
E2FormerV2Backbone = E2FormerBackbone

__all__ = [
    "E2former",
    "E2FormerBackbone",
    "E2FormerV2",
    "E2FormerV2Backbone",
]
