"""Capraz-modal tier: RGB referans <-> termal kare (ELoFTR / MatchAnything).

En zor katman (rapor 3.3). Hafif yontemler cozemiyor; ELoFTR (MatchAnything agirligi)
capraz-modaliteye ozel egitilmis tek offline secenek. VARSAYILAN KAPALI (config):
guven-yok payload'da bir FP dogrudan AP dusurur. Yuksek-guven eslesme esigi asilirsa acilir.
Lazy-load: yalnizca gercekten termal kare gelirse yuklenir (VRAM tasarrufu).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
import torch

from .config import Gorev3Config, DEFAULT


@dataclass
class CrossModalResult:
    bbox: Optional[Tuple[float, float, float, float]]
    inliers: int          # MAGSAC geometrik-tutarli eslesme sayisi (asil sinyal)
    n_matches: int        # ham ELoFTR eslesme sayisi (gurultu; tek basina kullanilmaz)


def is_thermalish(bgr: np.ndarray, tol: float = 6.0) -> bool:
    """Kare termal/tek-kanal benzeri mi? Kanallar arasi ortalama fark kucukse evet.

    Termal video BGR olarak gelse de R~=G~=B olur. Kesin degil; video_name ile de
    override edilebilir (bkz. reference_matcher).
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return True
    b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    d = (np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3.0
    return d < tol


class ELoFTRMatcher:
    def __init__(self, cfg: Gorev3Config = DEFAULT):
        cfg.apply_offline_env()
        self.cfg = cfg
        self.dev = cfg.device
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is None:
            from transformers import AutoModelForKeypointMatching, AutoImageProcessor
            self._proc = AutoImageProcessor.from_pretrained(self.cfg.eloftr_name)
            self._model = AutoModelForKeypointMatching.from_pretrained(
                self.cfg.eloftr_name).eval().to(self.dev)

    @torch.no_grad()
    def match(self, ref_bgr: np.ndarray, frame_bgr: np.ndarray) -> CrossModalResult:
        self._ensure()
        r_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self._proc(images=[[r_rgb, f_rgb]], return_tensors="pt").to(self.dev)
        out = self._model(**inp)
        sizes = torch.tensor(
            [[[r_rgb.shape[0], r_rgb.shape[1]], [f_rgb.shape[0], f_rgb.shape[1]]]],
            device=self.dev)  # cihaz uyumu: post_process kp'yi bu boyutlarla carpiyor
        res = self._proc.post_process_keypoint_matching(
            out, sizes, threshold=self.cfg.eloftr_score_thresh)[0]
        kp0 = res["keypoints0"].cpu().numpy()
        kp1 = res["keypoints1"].cpu().numpy()
        n = int(kp1.shape[0])
        if n < 4:
            return CrossModalResult(None, 0, n)
        # GEOMETRIK DOGRULAMA: ham eslesmeler capraz-modalitede kareye yayilmis gurultu
        # olabilir; MAGSAC homografi inlier'lari asil sinyaldir (matcher yoluyla ayni).
        H, mask = cv2.findHomography(kp0, kp1, cv2.USAC_MAGSAC, 5.0)
        inl = int(mask.sum()) if mask is not None else 0
        if inl < self.cfg.eloftr_min_inliers:
            return CrossModalResult(None, inl, n)
        fh, fw = frame_bgr.shape[:2]
        pts = kp1[mask.ravel().astype(bool)]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        x0 = float(np.clip(x0, 0, fw - 1)); y0 = float(np.clip(y0, 0, fh - 1))
        x1 = float(np.clip(x1, 0, fw - 1)); y1 = float(np.clip(y1, 0, fh - 1))
        af = ((x1 - x0) * (y1 - y0)) / float(fw * fh)
        if x1 <= x0 or y1 <= y0 or af > self.cfg.max_bbox_area_frac:
            return CrossModalResult(None, inl, n)
        return CrossModalResult((x0, y0, x1, y1), inl, n)
