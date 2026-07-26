"""Gorev 3 (Referans Nesne Tespiti) - merkezi konfigurasyon.

TUM esikler burada toplandi. Yarisma oncesi ornek videodan cikarilan mini-dogrulama
seti ile `tools/calibrate.py` uzerinden mAP@0.5'i maksimize edecek sekilde AYARLANMALIDIR.
Su anki degerler egitimli TAHMIN'dir (rapor Bolum 7.1). "# CALIBRATE" isaretli her
alan kalibrasyona tabidir.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import torch


@dataclass
class Gorev3Config:
    # --- Cihaz / offline ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    offline: bool = True                       # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE

    # --- Model isimleri (HF cache'te mevcut olmali) ---
    dinov2_name: str = "facebook/dinov2-with-registers-base"
    eloftr_name: str = "zju-community/matchanything_eloftr"

    # --- DINOv2 embedding ---
    dino_letterbox: int = 224                  # en-boy koruyan letterbox (rapor: kareye zorlama cos'u dusuruyor)
    dino_multi_scale: bool = True              # referansi birkac olcek/dondurme ile embed et (recall)

    # --- LightGlue matcher ---
    sp_max_keypoints: int = 1024
    magsac_thresh: float = 5.0                 # cv2.USAC_MAGSAC reprojection esigi (px)

    # --- Tier yonlendirme esikleri --- # CALIBRATE
    # Matcher scene-tier: ornek veride inlier'lar zayif (4-6) oldugu icin ateslenmiyor;
    # SAM+DINO halı sahayi zaten yakaliyor. Varsayilan KAPALI (kare basina 6x matcher
    # maliyetini onler). scene_min_inliers kalibre edilip ise yararsa acilabilir.
    use_matcher_scene_tier: bool = False
    scene_min_inliers: int = 40                # bunun ustu + genis kutu => SAHNE/ALAN tier (matcher kabul)
    scene_min_area_frac: float = 0.06          # kutu alani / kare alani; sahne icin buyuk beklenir
    compact_min_inliers: int = 8               # kompakt nesne icin matcher aday esigi
    samdino_min_cos: float = 0.45              # DINOv2 kosinus dogrulama esigi (kompakt/yer tier) # CALIBRATE

    # --- Gri (termal-stil) referans ozel yolu ---
    # RGB-egitimli DINOv2, gri referansta kararsiz: rotasyon augmentasyonu gri bicerdover
    # gorunumlerini beyaz binalara esliyor (rgb_mv gap -0.11). Cozum: gri referansi ve aday
    # segmentleri CLAHE-normalize GRI domainde + TEK-gorunum (rotasyonsuz) embed et.
    gray_ref_clahe: bool = True
    gray_ref_std_thresh: float = 12.0          # kanallar-arasi std bundan kucukse referans "gri" sayilir
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    samdino_min_cos_gray: float = 0.30         # gri referans icin kosinus esigi (ayri olcek) # CALIBRATE

    # --- Capraz-modal (termal kare + RGB referans) --- # CALIBRATE
    # BAGLI ve aktif, ama GEOMETRIK-DOGRULAMALI + precision-oncelikli. Ornek veride
    # (RGB-referans -> termal-kare, en zor hal) ELoFTR ham eslesmeleri MAGSAC'tan sonra
    # yalnizca 5-7 inlier'a dusuyor (gurultu) -> yuksek inlier esigi hicbir sey emitmiyor
    # (FP eklemez). Gercekten geometrik-tutarli bir eslesme (>=15 inlier) olursa yakalar.
    crossmodal_enabled: bool = True
    # Termal karede RGB-referans -> termal-kare (gercek capraz-modal) cozulemiyor (5-7 inlier);
    # yalnizca GRI/termal referanslar (termal<->termal, cozulebilir) icin ELoFTR calistir -> ~6x hiz.
    crossmodal_only_gray_refs: bool = True
    eloftr_min_inliers: int = 15               # MAGSAC inlier esigi (gurultuyu eler) # CALIBRATE
    eloftr_score_thresh: float = 0.2

    # --- Kutu / cikti temizligi ---
    bbox_pad_frac: float = 0.04                # inlier kutusuna kenar payi
    min_bbox_area_frac: float = 0.0005         # bundan kucuk kutulari ele (gurultu)
    max_bbox_area_frac: float = 0.6            # neredeyse tum kareyi kaplayan kutulari ele # CALIBRATE
    # FastSAM tum-kareyi-kaplayan arka plan segmenti gri FP dev-kutularin sebebiydi;
    # segmentasyon asamasinda bu buyuk segmentleri ADAY olmaktan cikar.
    seg_max_area_frac: float = 0.6
    # Segment secimi: argmax-kosinus bazen nesneyi iceren DEV arka-plan segmentini seciyor
    # (arka plan baskinligi). En yuksek kosinusun bu marji icindeki segmentler arasinda
    # EN SIKI (kucuk alanli) olani sec -> dev FP kutular yerine nesne kutusu. Sahne
    # referanslarini bozmaz (orada siki alternatif ayni kosinusu almaz). # CALIBRATE
    select_margin: float = 0.06
    one_box_per_reference: bool = True         # sartname 9.1.1: ayni nesneye 2. kutu AP'yi dusurur

    # --- FastSAM (opsiyonel; ultralytics kuruluysa kullanilir) ---
    use_fastsam_if_available: bool = True
    fastsam_weights: str = ""                  # bos ise gorev3/weights/FastSAM-s.pt denenir
    fastsam_imgsz: int = 768                   # SAM segment cozunurlugu (hiz<->kalite) # CALIBRATE

    # --- Kalibrasyon ---
    load_calibration: bool = True              # gorev3/calibrated.json varsa esikleri oradan al

    def __post_init__(self):
        if self.load_calibration:
            self._apply_calibration_file()

    def _apply_calibration_file(self) -> None:
        """tools/calibrate.py'nin urettigi calibrated.json'daki esikleri uygula (varsa)."""
        import json
        path = os.path.join(os.path.dirname(__file__), "calibrated.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        for key in ("samdino_min_cos", "samdino_min_cos_gray", "scene_min_inliers",
                    "eloftr_min_matches", "min_bbox_area_frac", "max_bbox_area_frac"):
            if key in data:
                setattr(self, key, data[key])

    def apply_offline_env(self) -> None:
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


DEFAULT = Gorev3Config()
