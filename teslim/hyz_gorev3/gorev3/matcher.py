"""SuperPoint + LightGlue + MAGSAC eslestirici.

Sahne/alan tier'inin birincil yontemi (rapor 3.2): referans<->kare yerel oznitelik
eslesmesi + MAGSAC geometrik tutarlilik. Kutu, referans kosolelerini homografi ile
kareye yansitarak uretilir (inlier bulutundan daha sikidir); homografi bozuksa
inlier disbukey kutusuna geri dusulur.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
import torch

from .config import Gorev3Config, DEFAULT


@dataclass
class MatchResult:
    bbox: Optional[Tuple[float, float, float, float]]  # (x0,y0,x1,y1) kare pikselinde
    inliers: int
    n_matches: int


def _gray_tensor(bgr: np.ndarray, dev: str) -> torch.Tensor:
    """SuperPoint (B,C,H,W) bekler -> (1,1,H,W)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(g)[None, None].float().to(dev) / 255.0


class LightGlueMatcher:
    def __init__(self, cfg: Gorev3Config = DEFAULT):
        from lightglue import LightGlue, SuperPoint
        self.cfg = cfg
        self.dev = cfg.device
        self.extractor = SuperPoint(max_num_keypoints=cfg.sp_max_keypoints).eval().to(self.dev)
        self.matcher = LightGlue(features="superpoint").eval().to(self.dev)

    @torch.no_grad()
    def extract(self, bgr: np.ndarray) -> dict:
        """Referans oznitelikleri onceden cikarilip cache'lenebilsin diye ayri."""
        return self.extractor({"image": _gray_tensor(bgr, self.dev)})

    @torch.no_grad()
    def match(self, ref_bgr: np.ndarray, frame_bgr: np.ndarray,
              ref_feats: Optional[dict] = None) -> MatchResult:
        from lightglue.utils import rbd
        f0 = ref_feats if ref_feats is not None else self.extract(ref_bgr)
        f1 = self.extractor({"image": _gray_tensor(frame_bgr, self.dev)})
        out = self.matcher({"image0": f0, "image1": f1})
        f0r, f1r, outr = rbd(f0), rbd(f1), rbd(out)
        m = outr["matches"]
        n = int(m.shape[0])
        if n < 4:
            return MatchResult(None, 0, n)
        kp0 = f0r["keypoints"][m[:, 0]].cpu().numpy()
        kp1 = f1r["keypoints"][m[:, 1]].cpu().numpy()
        H, mask = cv2.findHomography(kp0, kp1, cv2.USAC_MAGSAC, self.cfg.magsac_thresh)
        inl = int(mask.sum()) if mask is not None else 0
        if inl < 4:
            return MatchResult(None, inl, n)

        bbox = self._bbox_from_homography(H, ref_bgr.shape, frame_bgr.shape)
        if bbox is None:
            pts = kp1[mask.ravel().astype(bool)]
            bbox = self._bbox_from_points(pts, frame_bgr.shape)
        return MatchResult(bbox, inl, n)

    def _bbox_from_homography(self, H, ref_shape, frame_shape):
        if H is None:
            return None
        h, w = ref_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            return None
        bbox = self._bbox_from_points(proj, frame_shape)
        # dejenerasyon kontrolu: cok ince/buyuk kutu -> gecersiz say
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        fw, fh = frame_shape[1], frame_shape[0]
        area = (x1 - x0) * (y1 - y0)
        if area <= 0 or area > self.cfg.max_bbox_area_frac * fw * fh:
            return None
        return bbox

    def _bbox_from_points(self, pts, frame_shape):
        if pts is None or len(pts) == 0:
            return None
        fh, fw = frame_shape[:2]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        pad = self.cfg.bbox_pad_frac
        pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
        x0 = float(np.clip(x0, 0, fw - 1)); y0 = float(np.clip(y0, 0, fh - 1))
        x1 = float(np.clip(x1, 0, fw - 1)); y1 = float(np.clip(y1, 0, fh - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)
