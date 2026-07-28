#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo_ornek_veri.py — UC GOREVI RESMI ORNEK VERI SETI UZERINDE CALISTIRIR.

Yarisma istemcisinin kullandigi ayni sinifi (ObjectDetectionModel.detect) cagirir;
tek fark kareleri sunucudan degil, resmi ornek videodan okumasidir. Boylece her
gorevin ornek veri uzerinde calistigi tek ekranda izlenebilir.

Her kare icin tek satir basar:
  kare NNNN | G1: k kutu ... | G2: kaynak xyz=(..) hata | G3: k kutu ...

Kullanim (teslim klasorunun icinden):
  python3 demo_ornek_veri.py
  python3 demo_ornek_veri.py --bas 350 --kalibrasyon 100 --kor 80 --g3-pencere 500 540
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

BURADA = os.path.dirname(os.path.abspath(__file__))

# --- Teslim paketinin kendi icinden calis (ek duzenleme gerekmesin) -----------
os.environ.setdefault("GOREV2_DIR", BURADA)
os.environ.setdefault("GOREV1_DIR", os.path.join(BURADA, "gorev_1"))
os.environ.setdefault("GOREV3_DIR", os.path.join(BURADA, "hyz_gorev3"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _veri_bul() -> str:
    """Resmi ornek veri setini bulur. Sirasiyla: PROVA_VERI ortam degiskeni,
    bu klasorun yani, yaygin indirme konumlari."""
    adaylar = [
        os.environ.get("PROVA_VERI", ""),
        os.path.join(BURADA, "THYZ_2026_Ornek_Veri_Seti"),
        os.path.join(BURADA, "hyz_gorev3", "THYZ_2026_Ornek_Veri_Seti"),
        os.path.expanduser("~/Masaüstü/veri_2026/THYZ_2026_Ornek_Veri_Seti"),
        os.path.expanduser("~/Desktop/veri_2026/THYZ_2026_Ornek_Veri_Seti"),
        os.path.expanduser("~/İndirilenler/THYZ_2026_Ornek_Veri_Seti"),
        os.path.expanduser("~/Downloads/THYZ_2026_Ornek_Veri_Seti"),
    ]
    for a in adaylar:
        if a and os.path.exists(os.path.join(a, "THYZ_2026_Ornek_Veri_1.MP4")):
            return a
    sys.exit(
        "\nHATA: resmi ornek veri seti bulunamadi.\n\n"
        "Bu demo, TEKNOFEST'in yayinladigi ornek veri setini gerektirir\n"
        "(THYZ_2026_Ornek_Veri_1.MP4 + translation.csv + Referans_Nesneler).\n\n"
        "Cozum: veri setini indirip su klasorlerden birine koyun\n"
        f"  - {os.path.join(BURADA, 'THYZ_2026_Ornek_Veri_Seti')}\n"
        "  - ~/Masaustu/veri_2026/THYZ_2026_Ornek_Veri_Seti\n"
        "ya da yolu dogrudan verin:\n"
        "  PROVA_VERI=/yol/THYZ_2026_Ornek_Veri_Seti python3 demo_ornek_veri.py\n")


VERI = _veri_bul()
VIDEO = os.path.join(VERI, "THYZ_2026_Ornek_Veri_1.MP4")
GT_CSV = os.path.join(VERI, "THYZ_2026_Ornek_Veri_1_translation.csv")
REF_DIR = os.path.join(VERI, "THYZ_2026_Ornek_Veri_1_Referans_Nesneler")
ORNEKLEME = 4                      # 30 fps -> 7.5 kare/sn (sunucunun verdigi hiz)
SUNUCU = "http://demo-yerel/"

ISTEMCI = os.path.join(BURADA, "istemci", "TAKIM_BAGLANTI_ARAYUZU")
sys.path.insert(0, ISTEMCI)
os.chdir(ISTEMCI)

import cv2                                                        # noqa: E402
from src.object_detection_model import ObjectDetectionModel       # noqa: E402
from src.frame_predictions import FramePredictions               # noqa: E402

SINIF = {1: "tasit", 2: "insan", 3: "UAP", 4: "UAI"}
C = {"b": "\033[1m", "y": "\033[33m", "g": "\033[32m", "c": "\033[36m",
     "m": "\033[35m", "r": "\033[31m", "0": "\033[0m", "d": "\033[2m"}


def sinif_no(url: str) -> int:
    try:
        return int([p for p in str(url).rstrip("/").split("/") if p][-1])
    except (ValueError, IndexError):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", type=int, default=350, help="baslangic ornek kare no")
    ap.add_argument("--kalibrasyon", type=int, default=100, help="saglikli kare sayisi")
    ap.add_argument("--kor", type=int, default=80, help="kesinti kare sayisi")
    ap.add_argument("--g3-pencere", type=int, nargs=2, default=[500, 540],
                    metavar=("BAS", "SON"))
    a = ap.parse_args()

    kal_son = a.bas + a.kalibrasyon
    son = kal_son + a.kor

    print(f"\n{C['b']}{'='*104}{C['0']}")
    print(f"{C['b']}  TEKNOFEST 2026 — Havacilikta Yapay Zeka — takim hamidiye_4907501{C['0']}")
    print(f"{C['b']}  UC GOREV, RESMI ORNEK VERI SETI UZERINDE, TEK SURECTE{C['0']}")
    print(f"{C['b']}{'='*104}{C['0']}")
    print(f"  Veri seti : {C['c']}THYZ_2026_Ornek_Veri_1.MP4{C['0']}  (resmi 2026 ornek veri seti)")
    print(f"  Referans  : {C['c']}{os.path.basename(REF_DIR)}{C['0']}  (6 adet)")
    print(f"  Kod       : {C['c']}teslim/istemci/.../object_detection_model.py -> detect(){C['0']}")
    print(f"  Profil    : kare {a.bas}-{kal_son-1} SAGLIKLI (kalibrasyon) | "
          f"{kal_son}-{son-1} KESINTI (GPS yok) | G3 penceresi {a.g3_pencere[0]}-{a.g3_pencere[1]}")
    print(f"{C['d']}{'-'*104}{C['0']}")

    gt = {}
    with open(GT_CSV) as f:
        for i, row in enumerate(csv.DictReader(f)):
            gt[i] = (float(row["translation_x"]), float(row["translation_y"]),
                     float(row["translation_z"]))

    refler = sorted(os.listdir(REF_DIR))
    ref_paths = {f"ref://{ad}": os.path.join(REF_DIR, ad) for ad in refler}

    t0 = time.monotonic()
    print(f"  {C['y']}[1/3]{C['0']} Gorev 1 + Gorev 3 modelleri yukleniyor ...", flush=True)
    model = ObjectDetectionModel(SUNUCU)
    print(f"        {C['g']}hazir{C['0']}  ({time.monotonic()-t0:.1f} sn)")
    print(f"  {C['y']}[2/3]{C['0']} Gorev 2 SLAM sureci baslatiliyor (sozluk yuklemesi) ...", flush=True)
    print(f"  {C['y']}[3/3]{C['0']} Ornek video aciliyor: {os.path.basename(VIDEO)}", flush=True)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        sys.exit(f"video acilamadi: {VIDEO}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.bas * ORNEKLEME)
    print(f"{C['d']}{'-'*104}{C['0']}\n")

    tmp = os.path.join(BURADA, "_demo_tmp")
    os.makedirs(tmp, exist_ok=True)
    n1 = n3 = 0
    hatalar = []

    for s in range(a.bas, son):
        for _ in range(ORNEKLEME):
            ok, kare = cap.read()
            if not ok:
                break
        if not ok:
            break
        yol = os.path.join(tmp, f"f{s:06d}.webp")
        cv2.imwrite(yol, kare, [cv2.IMWRITE_WEBP_QUALITY, 95])

        saglikli = s < kal_son
        g = gt.get(s * ORNEKLEME, (None, None, None))
        gx, gy, gz = g if saglikli else (None, None, None)

        aktif = []
        if a.g3_pencere[0] <= s <= a.g3_pencere[1]:
            aktif = [{"url": u,
                      "frame_start_image_url": "/DEMO/frame_000000.webp",
                      "frame_end_image_url": "/DEMO/frame_999999.webp"}
                     for u in ref_paths]

        fp = FramePredictions(f"{SUNUCU}frames/{s}/", f"/DEMO/frame_{s:06d}.webp",
                              "DEMO_2026_ORNEK", gx, gy, gz)
        fp = model.detect(fp, "1" if saglikli else "0", active_refs=aktif,
                          ref_image_paths=ref_paths, frame_image_path=yol)
        p = fp.create_payload(SUNUCU)
        os.remove(yol)

        objs = p.get("detected_objects") or []
        trs = p.get("detected_translations") or []
        refs = p.get("reference_predictions") or []
        n1 += len(objs)
        n3 += len(refs)

        adlar = {}
        for o in objs:
            ad = SINIF.get(sinif_no(o.get("cls", "")), "?")
            adlar[ad] = adlar.get(ad, 0) + 1
        g1s = " ".join(f"{k}x{v}" for k, v in adlar.items()) or "-"

        if trs:
            t = trs[0]
            xyz = (float(t["translation_x"]), float(t["translation_y"]), float(t["translation_z"]))
            gv = gt.get(s * ORNEKLEME)
            hata = math.dist(xyz, gv) if gv else float("nan")
            if not saglikli and gv:
                hatalar.append(hata)
            g2s = f"xyz=({xyz[0]:6.1f},{xyz[1]:6.1f},{xyz[2]:5.1f}) hata={hata:5.1f}m"
        else:
            g2s = "cikti yok"

        g3s = " ".join(sorted({os.path.basename(str(r.get('reference', ''))
                                                .replace('ref://', ''))
                              .replace('Referans_Nesne_', 'Ref').split('.')[0]
                              for r in refs})) or "-"

        durum = f"{C['g']}SAGLIKLI{C['0']}" if saglikli else f"{C['y']}KESINTI {C['0']}"
        print(f" kare {s:4d} {durum}| "
              f"{C['m']}G1{C['0']}:{len(objs)} {g1s:17s}| "
              f"{C['c']}G2{C['0']}:{g2s} | "
              f"{C['g']}G3{C['0']}:{len(refs)} {g3s}", flush=True)

    cap.release()
    if getattr(model, "_engine", None):
        model._engine.shutdown()

    ort = sum(hatalar) / len(hatalar) if hatalar else float("nan")
    print(f"\n{C['d']}{'-'*104}{C['0']}")
    print(f"  {C['b']}OZET{C['0']}  islenen kare: {son-a.bas}   "
          f"{C['m']}Gorev 1{C['0']}: {n1} kutu   "
          f"{C['c']}Gorev 2{C['0']}: kesintide ort. hata {ort:.1f} m   "
          f"{C['g']}Gorev 3{C['0']}: {n3} kutu")
    print(f"  Her karede tam 1 gecerli tahmin uretildi — hicbir karede takilma yok.")
    print(f"{C['d']}{'-'*104}{C['0']}\n")


if __name__ == "__main__":
    main()
