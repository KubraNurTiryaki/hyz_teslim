#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gorev 3 demo — depodaki ornek KARELER uzerinde (video gerekmez).

gorev3/offline_test.py MP4 ister ve Windows yollari sabittir; bu betik ayni
ReferenceMatcher hattini repodaki test/ karelerine uygular.

Kullanim:
  python3 gorev3_kare_demo.py                 # RGB kareler + 6 referans
  python3 gorev3_kare_demo.py --termal        # termal kareler + termal referanslar
  python3 gorev3_kare_demo.py --kare 4        # ilk 4 kareyi isle
Cikti: _test_output/gorev3/ altina kutulu jpg + konsolda ozet.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2

BURADA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURADA)
from gorev3.config import Gorev3Config
from gorev3.reference_matcher import ReferenceMatcher

VERI = os.path.join(BURADA, "THYZ_2026_Ornek_Veri_Seti")
KARE_DIR = os.path.join(BURADA, "test")
CIKTI = os.path.join(BURADA, "_test_output", "gorev3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--termal", action="store_true", help="termal kareler + termal referanslar")
    ap.add_argument("--kare", type=int, default=6, help="islenecek kare sayisi")
    a = ap.parse_args()

    if a.termal:
        ref_dir = os.path.join(VERI, "THYZ_2026_Ornek_Veri_2_Termal_Referans_Nesneler")
        kareler = sorted(glob.glob(os.path.join(KARE_DIR, "termal_kare*.jpg")))
        etiket = "termal"
    else:
        ref_dir = os.path.join(VERI, "THYZ_2026_Ornek_Veri_1_Referans_Nesneler")
        kareler = sorted(glob.glob(os.path.join(KARE_DIR, "rgb_kare*.jpg")))
        etiket = "rgb"
    kareler = kareler[: a.kare]
    ref_paths = sorted(glob.glob(os.path.join(ref_dir, "Referans_Nesne_*")))
    if not ref_paths or not kareler:
        sys.exit(f"veri bulunamadi (ref={len(ref_paths)}, kare={len(kareler)})")

    os.makedirs(CIKTI, exist_ok=True)
    cfg = Gorev3Config()
    print(f"cihaz={cfg.device}  mod={etiket}  referans={len(ref_paths)}  kare={len(kareler)}")

    rm = ReferenceMatcher(cfg)
    t0 = time.time()
    for p in ref_paths:
        rm.register_reference(os.path.basename(p), p)
    print(f"referanslar kaydedildi ({time.time()-t0:.1f}s)")

    toplam_kabul = 0
    sureler = []
    for yol in kareler:
        frame = cv2.imread(yol)
        if frame is None:
            print(f"  {os.path.basename(yol)}: OKUNAMADI")
            continue
        t = time.time()
        drawn = frame.copy()
        hits = []
        ctx = rm.new_frame(frame, thermal=a.termal)
        for p in ref_paths:
            key = os.path.basename(p)
            bbox, det = rm.match_in(ctx, key)
            if bbox is None:
                continue
            toplam_kabul += 1
            hits.append((key, det.tier, det.confidence))
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 255, 0), 3)
            cv2.putText(drawn, f"{key[-6:]} {det.tier} {det.confidence:.2f}",
                        (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        dt = (time.time() - t) * 1000
        sureler.append(dt)
        durum = ", ".join(f"{k[-6:]}:{ti}/{c:.2f}" for k, ti, c in hits) or "- (kutu yok: FP korumasi)"
        print(f"  {os.path.basename(yol):<22} {dt:6.0f}ms/{len(ref_paths)}ref  tespit: {durum}")
        if hits:
            cv2.imwrite(os.path.join(CIKTI, f"{etiket}_{os.path.basename(yol)}"), drawn)

    if sureler:
        ort = sum(sureler) / len(sureler)
        print(f"\nozet: {len(sureler)} kare, {toplam_kabul} kabul edilen kutu, "
              f"ort {ort:.0f} ms/kare ({len(ref_paths)} referans icin)")
        print(f"gorseller: {CIKTI}")


if __name__ == "__main__":
    main()
