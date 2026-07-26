"""FastSAM + DINOv2 lokalizasyonu - rapor'un birincil "SAM+DINO" yontemi.

Neden birincil (bu veride ampirik olarak dogrulandi):
  - Matcher inlier'lari zayif (4-6) -> homografi kutusu kararsiz/gevsek.
  - DINOv2 TUM-referans embedding'i arka-plan baskinligindan muzdarip (cim her yeri
    esliyor). SAM once nesneyi segmentler -> embedding nesneye ozgu olur -> ayirt edici.

VERIMLILIK: segmentasyon + segment embedding'leri KARE BASINA BIR KEZ yapilir
(`segment_and_embed`), sonra her referans icin en yuksek kosinuslu segment secilir
(`best_for_reference`). Boylece N referans icin N kez SAM calismaz.

`ultralytics` kurulu olmali; agirlik: gorev3/weights/FastSAM-s.pt (offline cache).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import numpy as np

from .config import Gorev3Config, DEFAULT
from .embedder import DinoEmbedder

BBox = Tuple[float, float, float, float]


def fastsam_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class FrameSegments:
    boxes: np.ndarray      # (N,4) xyxy
    embs: np.ndarray       # (N,D) normalize DINOv2 gomme (RGB)
    crops: list = None     # segment crop'lari (gri-domain gomme lazy hesaplansin diye)
    embs_gray: np.ndarray = None   # (N,D) CLAHE-gri gomme (lazy; sadece gri referans varsa)


class SamDinoLocalizer:
    def __init__(self, cfg: Gorev3Config = DEFAULT, embedder: Optional[DinoEmbedder] = None):
        self.cfg = cfg
        self.embedder = embedder or DinoEmbedder(cfg)
        self._sam = None

    def _ensure(self):
        if self._sam is None:
            from ultralytics import FastSAM
            w = self.cfg.fastsam_weights or os.path.join(
                os.path.dirname(__file__), "weights", "FastSAM-s.pt")
            self._sam = FastSAM(w)

    def segment_and_embed(self, frame_bgr: np.ndarray,
                          imgsz: Optional[int] = None) -> FrameSegments:
        """Kare basina BIR kez: FastSAM segmentleri + her segmentin DINOv2 gommesi."""
        self._ensure()
        imgsz = imgsz or self.cfg.fastsam_imgsz
        res = self._sam(frame_bgr, device=self.cfg.device, retina_masks=True,
                        imgsz=imgsz, conf=0.4, iou=0.9, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return FrameSegments(np.zeros((0, 4), np.float32), np.zeros((0, 768), np.float32))
        boxes = res[0].boxes.xyxy.cpu().numpy()
        fh, fw = frame_bgr.shape[:2]
        frame_area = float(fw * fh)
        keep, crops = [], []
        for (x0, y0, x1, y1) in boxes:
            x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
            if x1i - x0i < 6 or y1i - y0i < 6:
                continue
            # tum-kareyi-kaplayan FastSAM arka plan segmentini ADAY olmaktan cikar
            if ((x1i - x0i) * (y1i - y0i)) / frame_area > self.cfg.seg_max_area_frac:
                continue
            crop = frame_bgr[max(0, y0i):y1i, max(0, x0i):x1i]
            if crop.size == 0:
                continue
            keep.append([float(x0), float(y0), float(x1), float(y1)])
            crops.append(crop)
        if not keep:
            return FrameSegments(np.zeros((0, 4), np.float32), np.zeros((0, 768), np.float32),
                                 crops=[])
        embs = self.embedder.embed_batch(crops)   # tek/parcali forward pass (hiz)
        del res                                    # ultralytics sonucu GPU tensor tutar; serbest birak
        return FrameSegments(np.asarray(keep, np.float32), embs, crops=crops)

    def ensure_gray(self, seg: FrameSegments) -> None:
        """Gri referans icin segmentlerin CLAHE-gri gommelerini (lazy) hesapla."""
        if seg.embs_gray is None:
            seg.embs_gray = self.embedder.embed_batch_clahe_gray(seg.crops or [])

    def best_for_reference(self, seg: FrameSegments, ref_emb: np.ndarray,
                           gray: bool = False) -> Tuple[Optional[BBox], float]:
        """Referans gommesine en cok benzeyen segment (siki kutu) + kosinus.

        gray=True: CLAHE-gri segment gommeleri kullanilir (gri referans yolu).
        """
        if seg.boxes.shape[0] == 0:
            return None, 0.0
        if gray:
            self.ensure_gray(seg)
            embs = seg.embs_gray
        else:
            embs = seg.embs
        if embs is None or embs.shape[0] == 0:
            return None, 0.0
        # ref_emb (K,D) cok-gorunum -> her segment icin max kosinus
        if ref_emb.ndim == 1:
            sims = embs @ ref_emb
        else:
            sims = (embs @ ref_emb.T).max(axis=1)
        max_cos = float(sims.max())
        # Siki-secim: en yuksek kosinusun `select_margin` marji icindeki segmentler
        # arasindan EN KUCUK ALANLI (en siki) olani sec. Argmax bazen nesneyi iceren dev
        # arka-plan segmentini seciyor; siki nesne segmenti genelde cok az dusuk skorlu.
        margin = self.cfg.select_margin
        cand = np.where(sims >= max_cos - margin)[0]
        b = seg.boxes
        areas = (b[cand, 2] - b[cand, 0]) * (b[cand, 3] - b[cand, 1])
        j = int(cand[int(np.argmin(areas))])
        return tuple(seg.boxes[j].tolist()), float(sims[j])

    # geriye donuk uyumluluk: tek cagrida segmentle+sec
    def localize(self, frame_bgr: np.ndarray, ref_emb: np.ndarray
                 ) -> Tuple[Optional[BBox], float]:
        return self.best_for_reference(self.segment_and_embed(frame_bgr), ref_emb)
