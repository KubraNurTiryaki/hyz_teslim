"""Offline uctan-uca test (ornek veri seti uzerinde).

Calistir:
    .venv\\Scripts\\python.exe -m gorev3.offline_test
veya
    .venv\\Scripts\\python.exe gorev3\\offline_test.py

Ne yapar:
  - HF_HUB_OFFLINE=1 ile tamamen cevrimdisi calisir (agirliklar cache'ten).
  - Video 1 (RGB) icin 6 referansi kaydeder, kareleri ornekler, her (kare,referans)
    icin ReferenceMatcher.match calistirir.
  - Kabul edilen tespitleri kutu+tier+guven ile _test_output/gorev3/ altina cizer.
  - Ozet tablo yazdirir. (GT olmadigi icin mAP hesaplanmaz; bu test sistemin uctan
    uca CALISTIGINI ve makul kutular urettigini dogrular. mAP kalibrasyonu ayri.)
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import glob
import time
import cv2

# paket olarak da, dosya olarak da calissin
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gorev3.config import Gorev3Config
    from gorev3.reference_matcher import ReferenceMatcher
else:
    from .config import Gorev3Config
    from .reference_matcher import ReferenceMatcher

DATA = r"C:\Users\Acer\Desktop\drone\hyz\THYZ_2026_Ornek_Veri_Seti"
OUT = r"C:\Users\Acer\Desktop\drone\hyz\_test_output\gorev3"
N_FRAMES = 8  # her videodan ornek kare sayisi


def sample_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames.append((fi, fr))
    cap.release()
    return frames, total


def run_video(tag, video_path, ref_dir, cfg, thermal):
    os.makedirs(OUT, exist_ok=True)
    ref_paths = sorted(glob.glob(os.path.join(ref_dir, "Referans_Nesne_*")))
    print(f"\n===== {tag} =====")
    print(f"video: {os.path.basename(video_path)}  referans: {len(ref_paths)} adet  thermal={thermal}")

    rm = ReferenceMatcher(cfg)
    t0 = time.time()
    for p in ref_paths:
        rm.register_reference(os.path.basename(p), p)
    print(f"referanslar kaydedildi ({time.time()-t0:.1f}s)")

    frames, total = sample_frames(video_path, N_FRAMES)
    print(f"toplam {total} kare; {len(frames)} kare ornekleniyor")

    n_accept = 0
    per_frame_ms = []
    for fi, frame in frames:
        t = time.time()
        drawn = frame.copy()
        hits = []
        ctx = rm.new_frame(frame, thermal=thermal)   # SAM bu kare icin en fazla 1 kez
        for p in ref_paths:
            key = os.path.basename(p)
            bbox, det = rm.match_in(ctx, key)
            if bbox is not None:
                n_accept += 1
                hits.append((key, det.tier, det.confidence, bbox))
                x0, y0, x1, y1 = [int(v) for v in bbox]
                cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 255, 0), 3)
                cv2.putText(drawn, f"{key[-6:]} {det.tier} {det.confidence:.2f}",
                            (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        dt = (time.time() - t) * 1000
        per_frame_ms.append(dt)
        status = ", ".join(f"{k[-6:]}:{ti}/{c:.2f}" for k, ti, c, _ in hits) or "-"
        print(f"  kare {fi:5d}  {dt:6.0f}ms/{len(ref_paths)}ref  tespit: {status}")
        if hits:
            outp = os.path.join(OUT, f"{tag}_frame{fi:05d}.jpg")
            cv2.imwrite(outp, drawn)
    avg = sum(per_frame_ms) / max(1, len(per_frame_ms))
    print(f"ozet: {n_accept} kabul; ort {avg:.0f}ms/kare ({len(ref_paths)} referans)  "
          f"~{avg/len(ref_paths):.0f}ms/referans")
    return n_accept


def main():
    cfg = Gorev3Config()
    print(f"cihaz: {cfg.device}  crossmodal_enabled: {cfg.crossmodal_enabled}")
    run_video("rgb", os.path.join(DATA, "THYZ_2026_Ornek_Veri_1.MP4"),
              os.path.join(DATA, "THYZ_2026_Ornek_Veri_1_Referans_Nesneler"), cfg, thermal=False)
    run_video("termal", os.path.join(DATA, "THYZ_2026_Ornek_Veri_2_Termal.MP4"),
              os.path.join(DATA, "THYZ_2026_Ornek_Veri_2_Termal_Referans_Nesneler"), cfg, thermal=True)
    print(f"\ngorseller: {OUT}")


if __name__ == "__main__":
    main()
