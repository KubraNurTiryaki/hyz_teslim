"""Gorsel test: kalibre sistemi videolar uzerinde calistir, bulunan referans nesneleri
kutulayarak `test/` klasorune kaydet.

Her tespit icin: renkli kutu + eslesEN referansin kucuk resmi + etiket (RefNN / tier / cos).
Yalnizca >=1 tespit olan kareler kaydedilir (kalabalik olmasin). Kalibre esikler kullanilir
(config calibrated.json'u otomatik okur).

Kullanim:
  .venv\\Scripts\\python.exe -m gorev3.tools.visualize
  .venv\\Scripts\\python.exe -m gorev3.tools.visualize --step 100 --out test
"""
from __future__ import annotations
import os, sys, glob, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gc
import cv2
import numpy as np
import torch
from gorev3.config import Gorev3Config
from gorev3.reference_matcher import ReferenceMatcher

DATA = r"C:\Users\Acer\Desktop\drone\hyz\THYZ_2026_Ornek_Veri_Seti"
PROJ = r"C:\Users\Acer\Desktop\drone\hyz"

# referans basina belirgin renk (BGR)
COLORS = [(60, 220, 60), (60, 160, 255), (255, 90, 90), (0, 235, 235),
          (230, 90, 230), (255, 200, 40)]


def draw_detection(img, box, ref_thumb, label, color):
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    # referans kucuk resmi kutunun sol-ust dışına
    th = cv2.resize(ref_thumb, (96, 96))
    ty0 = max(0, y0 - 100); ty1 = ty0 + 96
    tx0 = x0; tx1 = tx0 + 96
    if ty1 <= img.shape[0] and tx1 <= img.shape[1]:
        cv2.rectangle(img, (tx0 - 2, ty0 - 2), (tx1 + 2, ty1 + 2), color, 2)
        img[ty0:ty1, tx0:tx1] = th
    # etiket
    (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ly = max(thh + 4, y0 - 4)
    cv2.rectangle(img, (x0, ly - thh - 6), (x0 + tw + 6, ly + 2), color, -1)
    cv2.putText(img, label, (x0 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)


def run_video(tag, video, refdir, rm, refpaths, thumbs, out_dir, step, thermal):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n== {tag} == {os.path.basename(video)} ({total} kare, her {step}. kare)")
    saved = 0; scanned = 0
    fi = step
    while fi < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            fi += step; continue
        scanned += 1
        ctx = rm.new_frame(frame, thermal=thermal)
        drawn = frame.copy(); hits = []
        for i, p in enumerate(refpaths):
            key = os.path.basename(p)
            bbox, det = rm.match_in(ctx, key)
            if bbox is not None:
                num = key.split("_")[-1].split(".")[0]
                label = f"Ref{num} {det.tier} {det.confidence:.2f}"
                draw_detection(drawn, bbox, thumbs[key], label, COLORS[i % len(COLORS)])
                hits.append(f"Ref{num}({det.confidence:.2f})")
        if hits:
            cv2.putText(drawn, f"{tag} kare {fi}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            outp = os.path.join(out_dir, f"{tag}_kare{fi:05d}.jpg")
            cv2.imwrite(outp, drawn); saved += 1
            print(f"  kare {fi:5d}: {', '.join(hits)}", flush=True)
        # 4GB VRAM'de kare-kare birikimi onle: her karede GPU/CPU bellegini bosalt
        del ctx, drawn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fi += step
    cap.release()
    print(f"  -> {scanned} kare tarandi, {saved} tespitli kare kaydedildi")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=120, help="kac karede bir tara")
    ap.add_argument("--out", default=os.path.join(PROJ, "test"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Gorev3Config()
    print(f"cihaz={cfg.device} esikler: renkli={cfg.samdino_min_cos} gri={cfg.samdino_min_cos_gray} "
          f"crossmodal={cfg.crossmodal_enabled}")
    rm = ReferenceMatcher(cfg)

    total_saved = 0
    for tag, video, refdir, thermal in [
        ("rgb", os.path.join(DATA, "THYZ_2026_Ornek_Veri_1.MP4"),
         os.path.join(DATA, "THYZ_2026_Ornek_Veri_1_Referans_Nesneler"), False),
        ("termal", os.path.join(DATA, "THYZ_2026_Ornek_Veri_2_Termal.MP4"),
         os.path.join(DATA, "THYZ_2026_Ornek_Veri_2_Termal_Referans_Nesneler"), True),
    ]:
        refpaths = sorted(glob.glob(os.path.join(refdir, "Referans_Nesne_*")))
        thumbs = {os.path.basename(p): cv2.imread(p) for p in refpaths}
        # bu videonun referanslarini (temiz anahtarla) kaydet
        rm._refs.clear()
        for p in refpaths:
            rm.register_reference(os.path.basename(p), p)
        total_saved += run_video(tag, video, refdir, rm, refpaths, thumbs, args.out, args.step, thermal)

    print(f"\nTOPLAM {total_saved} tespitli kare -> {args.out}")


if __name__ == "__main__":
    main()
