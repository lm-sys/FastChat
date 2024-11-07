from dataclasses import dataclass, field
from typing import List


@dataclass
class Context:
    text_models: List[str] = field(default_factory=list)
    all_text_models: List[str] = field(default_factory=list)
    vision_models: List[str] = field(default_factory=list)
    all_vision_models: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    all_models: List[str] = field(default_factory=list)
