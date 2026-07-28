#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gorev2_duman_testi.py — Görev 2 motorunun VERİ SETİ OLMADAN sağlık kontrolü.

Tam prova (resmi_mock.py / canli_prova.sh) 2025 oturum kareleri ister; bu betik
onlar indirilmeden de motorun ayakta olduğunu doğrular:

  1. SLAM süreci (mono_folder_watch) başlıyor ve READY veriyor mu?
  2. sağlık=1 karede referans AYNEN geri dönüyor mu? (kaynak=echo, hata=0)
  3. sağlık=0 karede — SLAM izleme kuramasa bile — motor NaN'sız bir konum
     üretiyor mu? (fail-safe: kaynak=deadreckon; "her kareye tam 1 tahmin")

Kullanım:
  python3 gorev2_duman_testi.py [--settings <yaml>] [--kare 12]
Varsayılan yaml: ~/SP_SLAM3/Examples/Monocular/teknofest_1080p.yaml
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

BURADA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURADA)
from gorev2_engine import Gorev2Engine

VARSAYILAN_YAML = os.path.expanduser(
    "~/SP_SLAM3/Examples/Monocular/teknofest_1080p.yaml")
# Veri seti yoksa elde ne varsa onu kullan (hyz deposundaki örnek kareler)
KARE_KAYNAKLARI = [
    os.environ.get("DUMAN_KARELERI", ""),                       # elle yol verilebilir
    os.path.join(BURADA, "hyz_gorev3", "test", "rgb_kare*.jpg"),  # paket icinden
    os.path.join(BURADA, "yarisma_kareler", "kare_*.jpg"),        # paket icinden
    os.path.expanduser("~/Masaüstü/hyz_gorev3/test/rgb_kare*.jpg"),
    os.path.expanduser("~/Desktop/hyz_gorev3/test/rgb_kare*.jpg"),
]


def kareleri_bul():
    for desen in KARE_KAYNAKLARI:
        if not desen:
            continue
        yollar = sorted(glob.glob(desen))
        if yollar:
            return yollar
    sys.exit(
        "\nHATA: duman testi icin ornek kare bulunamadi.\n\n"
        "Bu test, birkac ardisik ucus karesi gerektirir (icerik onemli degil).\n"
        "Cozum — sunlardan biri:\n"
        f"  - hyz deposunun test klasorunu su yola kopyalayin:\n"
        f"      {os.path.join(BURADA, 'hyz_gorev3', 'test')}\n"
        "  - ya da kendi karelerinizin yolunu verin:\n"
        "      DUMAN_KARELERI='/yol/kareler/*.jpg' python3 gorev2_duman_testi.py\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=VARSAYILAN_YAML)
    ap.add_argument("--kare", type=int, default=12)
    ap.add_argument("--run-dir", default=os.path.join(BURADA, "run_duman_testi"))
    a = ap.parse_args()

    if not os.path.exists(a.settings):
        sys.exit(f"HATA: kalibrasyon yaml yok: {a.settings}\n"
                 f"SP_SLAM3 kurulu mu? (KURULUM.md §5)")
    yollar = kareleri_bul()
    if not yollar:
        sys.exit("HATA: test karesi bulunamadı (hyz_gorev3/test/ veya yarisma_kareler/)")
    yollar = (yollar * ((a.kare // len(yollar)) + 1))[: a.kare]

    print(f"yaml     : {a.settings}")
    print(f"kare     : {len(yollar)} adet")
    print(f"run-dir  : {a.run_dir}")

    eng = Gorev2Engine(settings=a.settings, run_dir=a.run_dir)
    eng.start()
    print("SLAM süreci başladı (READY).")

    hatalar = []
    kaynaklar = {}
    try:
        for i, yol in enumerate(yollar):
            img = cv2.imread(yol)
            saglik = 1 if i < len(yollar) // 2 else 0        # yarısı sağlıklı, yarısı kesinti
            ref = np.array([i * 1.0, i * 0.5, -i * 0.2]) if saglik else None
            xyz, kaynak = eng.process_frame(i, img, ref, saglik)
            kaynaklar[kaynak] = kaynaklar.get(kaynak, 0) + 1
            if xyz is None or not np.all(np.isfinite(xyz)):
                hatalar.append(f"kare {i}: GEÇERSİZ çıktı {xyz}")
                continue
            if saglik == 1:
                sapma = float(np.linalg.norm(np.asarray(xyz) - ref))
                if sapma > 1e-6:
                    hatalar.append(f"kare {i}: sağlık=1 yankı hatası {sapma:.3e} m")
            print(f"  kare {i:3d} sağlık={saglik} kaynak={kaynak:<11} "
                  f"xyz=({xyz[0]:8.3f},{xyz[1]:8.3f},{xyz[2]:8.3f})")
    finally:
        eng.shutdown()

    print(f"\nkaynak dağılımı: {kaynaklar}")
    if hatalar:
        print("BAŞARISIZ:")
        for h in hatalar:
            print("  -", h)
        sys.exit(1)
    print("GÖREV 2 DUMAN TESTİ TAMAM — her karede geçerli (NaN'sız) tahmin üretildi,\n"
          "sağlık=1 karelerde yankı hatası tam 0.")


if __name__ == "__main__":
    main()
