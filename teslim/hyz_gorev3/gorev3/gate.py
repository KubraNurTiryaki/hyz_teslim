"""Cikti kapisi (gate): esikleme + kutu temizligi + TEK-KUTU kurali.

mAP'i asil belirleyen katman. Gerekce:
- Payload'da guven skoru YOK -> her gonderilen kutu KESIN pozitif sayilir. Emin
  olunmayan tespit HIC gonderilmemeli (None dondur). FP dogrudan AP dusurur.
- Sartname 9.1.1: ayni referansa 2. kutu AP'yi dusurur -> referans basina en fazla 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import Gorev3Config, DEFAULT

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    bbox: Optional[BBox]
    confidence: float          # tier'e gore normalize skor (kosinus veya inlier-tabanli)
    tier: str                  # "scene" | "compact" | "crossmodal"
    accepted: bool


def sanitize_bbox(bbox: Optional[BBox], frame_w: int, frame_h: int,
                  cfg: Gorev3Config = DEFAULT) -> Optional[BBox]:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, frame_w - 1)); x1 = max(0.0, min(x1, frame_w - 1))
    y0 = max(0.0, min(y0, frame_h - 1)); y1 = max(0.0, min(y1, frame_h - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    area_frac = ((x1 - x0) * (y1 - y0)) / float(frame_w * frame_h)
    if area_frac < cfg.min_bbox_area_frac or area_frac > cfg.max_bbox_area_frac:
        return None
    return (x0, y0, x1, y1)


def finalize(det: Detection, frame_w: int, frame_h: int,
             cfg: Gorev3Config = DEFAULT) -> Optional[BBox]:
    """Kabul edilmemis veya gecersiz kutu -> None (kare icin hicbir sey gonderme)."""
    if not det.accepted or det.bbox is None:
        return None
    return sanitize_bbox(det.bbox, frame_w, frame_h, cfg)
