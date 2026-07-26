"""Gorev 3 - Referans Nesne Tespiti (hibrit sistem).

Kullanim (yarisma dongusunde):
    from gorev3 import ReferenceMatcher
    rm = ReferenceMatcher()
    rm.register_reference(ref_url, ref_image_path)   # oturum basinda her referans icin
    bbox, det = rm.match(frame_bgr, ref_url)          # her kare icin; bbox None ise gonderme
"""
from .config import Gorev3Config, DEFAULT
from .reference_matcher import ReferenceMatcher
from .gate import Detection

__all__ = ["ReferenceMatcher", "Gorev3Config", "DEFAULT", "Detection"]
