"""DINOv2 gomme (embedding) - kompakt nesne tier'inin dogal guven kaynagi.

Payload'da guven skoru YOK; bu yuzden DINOv2 kosinusu, bir tespitin gercekten
referans olup olmadigini eslemenin (FP koruma) tek dogal yoludur. Referans crop'u
letterbox ile embed edilir (rapor: kareye zorlama en-boy oranini bozup cos'u dusuruyor).
"""
from __future__ import annotations
import numpy as np
import cv2
import torch

from .config import Gorev3Config, DEFAULT

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
    """En-boy oranini koruyarak size x size tuvale yerlestir (siyah dolgu)."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def is_grayscale_image(bgr: np.ndarray, std_thresh: float = 12.0) -> bool:
    """Goruntu (BGR gelse de) gri/tek-kanal benzeri mi? Kanallar-arasi fark kucukse evet."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return True
    b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    d = (np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3.0
    return d < std_thresh


class DinoEmbedder:
    def __init__(self, cfg: Gorev3Config = DEFAULT):
        cfg.apply_offline_env()
        from transformers import AutoModel
        self.cfg = cfg
        self.dev = cfg.device
        self.model = AutoModel.from_pretrained(cfg.dinov2_name).eval().to(self.dev)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.dev).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.dev).view(1, 3, 1, 1)
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                      tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))

    def _clahe_gray3(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> CLAHE-normalize gri -> 3 kanala kopyala (gri-domain eslesme icin)."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(self._clahe.apply(g), cv2.COLOR_GRAY2BGR)

    @torch.no_grad()
    def _embed_one(self, bgr: np.ndarray) -> np.ndarray:
        img = letterbox(bgr, self.cfg.dino_letterbox)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.dev)
        t = (t - self._mean) / self._std
        out = self.model(pixel_values=t)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        v = pooled[0].float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-9)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """Tek bir gomme (kare crop'lari icin)."""
        return self._embed_one(bgr)

    @torch.no_grad()
    def embed_batch(self, bgr_list, batch_size: int = 8) -> np.ndarray:
        """Cok sayida crop'u tek forward pass'lerde gom (SAM segmentleri icin - hiz).

        Returns (N, D) normalize gomme matrisi. VRAM icin batch_size ile parcalanir.
        """
        if not bgr_list:
            return np.zeros((0, 768), np.float32)
        out = []
        S = self.cfg.dino_letterbox
        for i in range(0, len(bgr_list), batch_size):
            chunk = bgr_list[i:i + batch_size]
            arr = np.stack([
                cv2.cvtColor(letterbox(c, S), cv2.COLOR_BGR2RGB) for c in chunk
            ]).astype(np.float32) / 255.0                       # (B,H,W,3)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.dev)
            t = (t - self._mean) / self._std
            o = self.model(pixel_values=t)
            pooled = o.pooler_output if getattr(o, "pooler_output", None) is not None \
                else o.last_hidden_state[:, 0]
            v = pooled.float().cpu().numpy()
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            out.append(v)
        return np.concatenate(out, 0)

    def embed_clahe_gray(self, bgr: np.ndarray) -> np.ndarray:
        """Gri referans/crop icin CLAHE-gri, TEK-gorunum (rotasyonsuz) gomme.

        Rotasyon augmentasyonu gri referansta yanlis-pozitif uretiyordu (deney: rgb_mv gap
        -0.11 -> cl_sv gap +0.05); bu yuzden gri yolda rotasyon YOK.
        """
        return self._embed_one(self._clahe_gray3(bgr))

    def embed_batch_clahe_gray(self, bgr_list, batch_size: int = 24) -> np.ndarray:
        """Cok sayida crop'u CLAHE-gri domainde gom (SAM segmentleri icin)."""
        if not bgr_list:
            return np.zeros((0, 768), np.float32)
        return self.embed_batch([self._clahe_gray3(c) for c in bgr_list], batch_size)

    def embed_reference(self, bgr: np.ndarray) -> np.ndarray:
        """Referans icin cok-olcek/dondurme gomme kumesi (recall: acili havadan cekim).

        Donen matris (K, D); crop dogrulamasi max kosinus alacak sekilde kullanilir.
        """
        views = [bgr]
        if self.cfg.dino_multi_scale:
            for ang in (90, 180, 270):
                views.append(cv2.rotate(bgr, {90: cv2.ROTATE_90_CLOCKWISE,
                                              180: cv2.ROTATE_180,
                                              270: cv2.ROTATE_90_COUNTERCLOCKWISE}[ang]))
        return np.stack([self._embed_one(v) for v in views], axis=0)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """a: (D,) crop gommesi; b: (K,D) veya (D,) referans gomme(leri) -> max kosinus."""
        if b.ndim == 1:
            return float(a @ b)
        return float(np.max(b @ a))
