from dataclasses import dataclass
from typing import Optional, Tuple


FeatureVector = Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class TVS:
    axis_3: float
    axis_7: float
    axis_14: float


@dataclass(frozen=True)
class ExperienceEntry:
    object_id: str
    tvs: TVS


@dataclass(frozen=True)
class Interpretation:
    agent_id: str
    mode: str
    meaning_label: str
    feature_vector: FeatureVector
    memory_reference: Optional[str] = None
    notes: Optional[str] = None
