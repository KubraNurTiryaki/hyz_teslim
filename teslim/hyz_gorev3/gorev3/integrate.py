"""Resmi arayuze (TAKIM_BAGLANTI_ARAYUZU) baglanti kancasi.

Resmi `object_detection_model.py` icindeki Gorev-3 dongusu su an sabit bir bbox
donuyor (placeholder). Asagidaki `ReferenceObjectDetector`, o placeholder'i gercek
hibrit sistemle degistirir. Entegrasyon:

    # object_detection_model.py -> ObjectDetectionModel.__init__ icinde:
    from gorev3.integrate import ReferenceObjectDetector
    self.ref_detector = ReferenceObjectDetector()

    # detect() icindeki Gorev-3 dongusu su hale gelir:
    for ref in (active_refs or []):
        start_img = ref.get('frame_start_image_url', '')
        end_img = ref.get('frame_end_image_url', '')
        if not (start_img and end_img and start_img <= prediction.image_url <= end_img):
            continue
        bbox = self.ref_detector.detect_for_frame(
            frame_image_path, ref['url'], ref_image_paths.get(ref['url']),
            video_name=prediction.video_name)
        if bbox is None:        # emin degilsek HIC kutu gonderme (FP koruma)
            continue
        prediction.add_reference_prediction(
            ReferencePrediction(ref['url'], prediction.frame_url, *bbox))
"""
from __future__ import annotations
from typing import Optional, Tuple
import logging
import cv2

from .config import Gorev3Config, DEFAULT
from .reference_matcher import ReferenceMatcher

BBox = Tuple[float, float, float, float]


class ReferenceObjectDetector:
    def __init__(self, cfg: Gorev3Config = DEFAULT):
        self.cfg = cfg
        self.rm = ReferenceMatcher(cfg)
        self._registered = set()
        # Ayni kare birden cok referans icin sirayla cagrilir; SAM segmentasyonu
        # kare basina bir kez olsun diye kare baglamini yol'a gore cache'liyoruz.
        self._frame_key = None
        self._frame_ctx = None

    def _ensure_registered(self, ref_url: str, ref_image_path: Optional[str]) -> bool:
        if ref_url in self._registered:
            return True
        if not ref_image_path:
            logging.warning(f"[gorev3] referans goruntu yolu yok: {ref_url}")
            return False
        try:
            self.rm.register_reference(ref_url, ref_image_path)
            self._registered.add(ref_url)
            return True
        except Exception as e:
            logging.error(f"[gorev3] referans kaydi basarisiz {ref_url}: {e}")
            return False

    def detect_for_frame(self, frame_image_path: str, ref_url: str,
                         ref_image_path: Optional[str],
                         video_name: Optional[str] = None) -> Optional[BBox]:
        """Tek kare + tek referans -> gonderilecek bbox | None."""
        if not self._ensure_registered(ref_url, ref_image_path):
            return None
        thermal = None
        if video_name and ("termal" in video_name.lower() or "thermal" in video_name.lower()):
            thermal = True
        # kare baglami cache: ayni frame_image_path icin bir kez olustur
        if self._frame_key != frame_image_path:
            frame = cv2.imread(frame_image_path)
            if frame is None:
                logging.error(f"[gorev3] kare okunamadi: {frame_image_path}")
                return None
            self._frame_key = frame_image_path
            self._frame_ctx = self.rm.new_frame(frame, thermal=thermal)
        try:
            bbox, det = self.rm.match_in(self._frame_ctx, ref_url)
            if bbox is not None:
                logging.info(f"[gorev3] {ref_url} tier={det.tier} conf={det.confidence:.3f} bbox={bbox}")
            return bbox
        except Exception as e:
            logging.error(f"[gorev3] match hatasi {ref_url}: {e}")
            return None
