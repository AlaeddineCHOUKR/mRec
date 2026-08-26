"""Model registry. A name in a config resolves to exactly one class here."""

from __future__ import annotations

from mrec.eval.runner import Recommender
from mrec.models.actr import MODELS as _ACTR_MODELS
from mrec.models.genrec_recommender import MODELS as _GENREC_MODELS
from mrec.models.pisa_recommender import MODELS as _PISA_MODELS
from mrec.models.topfreq import MODELS as _FREQ_MODELS

MODELS: dict[str, type] = {**_FREQ_MODELS, **_ACTR_MODELS, **_PISA_MODELS, **_GENREC_MODELS}


def build_model(name: str, **params) -> Recommender:
    if name not in MODELS:
        raise ValueError(f"unknown model `{name}`; known: {sorted(MODELS)}")
    return MODELS[name](**params)
