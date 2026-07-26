"""ReferenceMatcher - hibrit orkestrasyon (rapor Bolum 5).

Kare-seviyesi akis (verimli): her kare icin `new_frame()` bir baglam olusturur;
SAM segmentasyonu (varsa) o kare icin BIR KEZ yapilir ve tum aktif referanslar
paylasir. `match_in(ctx, ref_key)` her referans icin bbox+guven | None dondurur.

Yonlendirme (referans tipine gore):
  - Termal kare -> crossmodal tier (ELoFTR) [config'de varsayilan kapali]
  - RGB: (1) matcher buyuk+yuksek-inlier kutu verirse SAHNE/ALAN tier kabul;
         (2) aksi halde SAM+DINO (FastSAM segment -> DINOv2 kosinus) birincil;
         (3) SAM yoksa matcher adayi + DINOv2 crop dogrulamasi (fallback).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from .config import Gorev3Config, DEFAULT
from .embedder import DinoEmbedder, is_grayscale_image
from .matcher import LightGlueMatcher
from .crossmodal import ELoFTRMatcher, is_thermalish
from .gate import Detection, finalize
from . import localizer_samdino as sd

BBox = Tuple[float, float, float, float]


@dataclass
class _RefCache:
    bgr: np.ndarray
    feats: dict
    emb: np.ndarray             # (K,D) cok-gorunumlu DINOv2 gomme (RGB)
    gray_img: bool = False      # referans goruntu gri/termal mi (CLAHE bayragindan bagimsiz)
    is_gray: bool = False       # gri CLAHE yolu aktif mi (gray_ref_clahe AND gray_img)
    emb_gray: np.ndarray = None # (D,) CLAHE-gri tek-gorunum gomme (gri referans yolu)


@dataclass
class FrameCtx:
    frame: np.ndarray
    thermal: bool
    segments: Optional["sd.FrameSegments"] = None   # lazy: ilk RGB referansta doldurulur
    _seg_done: bool = False


class ReferenceMatcher:
    def __init__(self, cfg: Gorev3Config = DEFAULT):
        cfg.apply_offline_env()
        self.cfg = cfg
        self.embedder = DinoEmbedder(cfg)
        self.matcher = LightGlueMatcher(cfg)
        self._cross: Optional[ELoFTRMatcher] = None
        self._samdino: Optional[sd.SamDinoLocalizer] = None
        self._refs: Dict[str, _RefCache] = {}
        self._use_sam = cfg.use_fastsam_if_available and sd.fastsam_available()

    # ---- referans kaydi (oturum basinda bir kez) ----
    def register_reference(self, ref_key: str, ref_path_or_bgr) -> None:
        bgr = ref_path_or_bgr if isinstance(ref_path_or_bgr, np.ndarray) else cv2.imread(ref_path_or_bgr)
        if bgr is None:
            raise FileNotFoundError(f"Referans okunamadi: {ref_path_or_bgr}")
        gray_img = is_grayscale_image(bgr, self.cfg.gray_ref_std_thresh)
        is_gray = self.cfg.gray_ref_clahe and gray_img
        self._refs[ref_key] = _RefCache(
            bgr=bgr,
            feats=self.matcher.extract(bgr),
            emb=self.embedder.embed_reference(bgr),
            gray_img=gray_img,
            is_gray=is_gray,
            # gri referans: CLAHE-gri TEK-gorunum (rotasyonsuz) gomme
            emb_gray=self.embedder.embed_clahe_gray(bgr) if is_gray else None,
        )

    def _sam_localizer(self) -> sd.SamDinoLocalizer:
        if self._samdino is None:
            self._samdino = sd.SamDinoLocalizer(self.cfg, self.embedder)
        return self._samdino

    def _cross_matcher(self) -> ELoFTRMatcher:
        if self._cross is None:
            self._cross = ELoFTRMatcher(self.cfg)
        return self._cross

    # ---- kare baglami ----
    def new_frame(self, frame_bgr: np.ndarray, thermal: Optional[bool] = None) -> FrameCtx:
        is_therm = is_thermalish(frame_bgr) if thermal is None else thermal
        return FrameCtx(frame=frame_bgr, thermal=is_therm)

    def _ensure_segments(self, ctx: FrameCtx) -> "sd.FrameSegments":
        if not ctx._seg_done:
            ctx.segments = self._sam_localizer().segment_and_embed(ctx.frame)
            ctx._seg_done = True
        return ctx.segments

    # ---- referans eslesmesi (kare baglami icinde) ----
    def match_in(self, ctx: FrameCtx, ref_key: str) -> Tuple[Optional[BBox], Detection]:
        if ref_key not in self._refs:
            raise KeyError(f"Referans kayitli degil: {ref_key}")
        ref = self._refs[ref_key]
        fh, fw = ctx.frame.shape[:2]

        # --- Capraz-modal (termal) ---
        if ctx.thermal:
            skip = (not self.cfg.crossmodal_enabled
                    or (self.cfg.crossmodal_only_gray_refs and not ref.gray_img))
            if skip:  # RGB-referans -> termal-kare cozulemiyor; bosuna ELoFTR calistirma
                return None, Detection(None, 0.0, "crossmodal", accepted=False)
            cm = self._cross_matcher().match(ref.bgr, ctx.frame)
            conf = min(1.0, cm.inliers / (2.0 * self.cfg.eloftr_min_inliers))
            det = Detection(cm.bbox, conf, "crossmodal", accepted=cm.bbox is not None)
            return finalize(det, fw, fh, self.cfg), det

        # --- RGB (1): matcher sahne/alan tier'i (opsiyonel; varsayilan kapali) ---
        if self.cfg.use_matcher_scene_tier:
            mr = self.matcher.match(ref.bgr, ctx.frame, ref_feats=ref.feats)
            if mr.bbox is not None and mr.inliers >= self.cfg.scene_min_inliers:
                x0, y0, x1, y1 = mr.bbox
                area_frac = ((x1 - x0) * (y1 - y0)) / float(fw * fh)
                if area_frac >= self.cfg.scene_min_area_frac:
                    conf = min(1.0, mr.inliers / (2.0 * self.cfg.scene_min_inliers))
                    det = Detection(mr.bbox, conf, "scene", accepted=True)
                    return finalize(det, fw, fh, self.cfg), det

        # --- RGB (2): SAM+DINO (birincil, bu veride en guclu) ---
        if self._use_sam:
            seg = self._ensure_segments(ctx)
            if ref.is_gray:   # gri referans: CLAHE-gri domain + ayri esik (ref04 sorunu)
                box, cos = self._sam_localizer().best_for_reference(seg, ref.emb_gray, gray=True)
                thr, tier = self.cfg.samdino_min_cos_gray, "gray"
            else:
                box, cos = self._sam_localizer().best_for_reference(seg, ref.emb)
                thr, tier = self.cfg.samdino_min_cos, "compact"
            accepted = box is not None and cos >= thr
            det = Detection(box, cos, tier, accepted=accepted)
            return finalize(det, fw, fh, self.cfg), det

        # --- RGB (3): SAM yoksa matcher adayi + DINOv2 crop dogrulamasi (fallback) ---
        mr = self.matcher.match(ref.bgr, ctx.frame, ref_feats=ref.feats)
        if mr.bbox is not None and mr.inliers >= self.cfg.compact_min_inliers:
            cos = self._verify_crop(ctx.frame, mr.bbox, ref.emb)
            accepted = cos >= self.cfg.samdino_min_cos
            det = Detection(mr.bbox, cos, "compact", accepted=accepted)
            return finalize(det, fw, fh, self.cfg), det

        return None, Detection(mr.bbox, 0.0, "compact", accepted=False)

    # ---- tek-cagri kolaylik (bir kare + bir referans) ----
    def match(self, frame_bgr: np.ndarray, ref_key: str,
              thermal: Optional[bool] = None) -> Tuple[Optional[BBox], Detection]:
        return self.match_in(self.new_frame(frame_bgr, thermal), ref_key)

    # ---- KALIBRASYON: esikten BAGIMSIZ ham aday (box, skor, tier) ----
    def candidate(self, ctx: FrameCtx, ref_key: str) -> Tuple[Optional[BBox], float, str]:
        """En iyi aday kutu + ham skoru (esik uygulanmadan). calibrate.py bunu kare
        basina 1 kez toplayip esikleri offline tarar (SAM'i tekrar calistirmadan).

        Skor anlami tier'e gore: compact/scene -> kosinus; crossmodal -> eslesme sayisi.
        """
        if ref_key not in self._refs:
            raise KeyError(f"Referans kayitli degil: {ref_key}")
        ref = self._refs[ref_key]
        if ctx.thermal:
            if (not self.cfg.crossmodal_enabled
                    or (self.cfg.crossmodal_only_gray_refs and not ref.gray_img)):
                return None, 0.0, "crossmodal"
            cm = self._cross_matcher().match(ref.bgr, ctx.frame)
            return cm.bbox, float(cm.inliers), "crossmodal"
        if self._use_sam:
            seg = self._ensure_segments(ctx)
            if ref.is_gray:
                box, cos = self._sam_localizer().best_for_reference(seg, ref.emb_gray, gray=True)
                return box, cos, "gray"
            box, cos = self._sam_localizer().best_for_reference(seg, ref.emb)
            return box, cos, "compact"
        mr = self.matcher.match(ref.bgr, ctx.frame, ref_feats=ref.feats)
        if mr.bbox is None:
            return None, 0.0, "compact"
        cos = self._verify_crop(ctx.frame, mr.bbox, ref.emb)
        return mr.bbox, cos, "compact"

    def _verify_crop(self, frame_bgr, bbox, ref_emb) -> float:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0 = max(0, x0); y0 = max(0, y0)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return 0.0
        return self.embedder.cosine(self.embedder.embed(crop), ref_emb)
