from dataclasses import dataclass, field


@dataclass
class TagResult:
    phrase:   str
    tags:     list[str] = field(default_factory=list)
    motscles: list[str] = field(default_factory=list)
