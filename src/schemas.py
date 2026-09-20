from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Segment:
    """Represents a speech segment detected by VAD or reference annotations."""
    start_s: float
    end_s: float
    source: str = "webrtc"
    confidence: Optional[float] = 1.0

    @property
    def duration(self) -> float:
        """Returns segment duration in seconds."""
        return max(0.0, round(self.end_s - self.start_s, 6))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes segment to dictionary."""
        return {
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration": round(self.duration, 4),
            "source": self.source,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Constructs Segment instance from dictionary."""
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            source=str(data.get("source", "webrtc")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else 1.0,
        )
