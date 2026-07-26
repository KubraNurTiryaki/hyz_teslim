#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Yerel prova için sentetik kare seti + GT üretir (gerçek veri seti yokken).

resmi_mock.py 'frame_%06d.webp' adlandırması ve 'frame_id,x,y,z' başlıklı GT bekler.
Elimizdeki 17 gerçek örnek kare döngüsel tekrarlanarak N kare üretilir; GT ise
2026 örnek veri setinin gerçek translation.csv'sinden ilk N satır alınır.

DİKKAT: Bu bir TESİSAT (plumbing) testidir — kareler ardışık bir uçuş değil,
dolayısıyla SLAM izlemesi anlamlı değildir. Amaç: istemci↔sunucu protokolü,
üç görevin tek süreçte koşması, payload serileştirmesi ve gönderim döngüsü.
"""
import csv
import os
import sys

import cv2

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prova_yerel")
os.makedirs(SC, exist_ok=True)
KAYNAK = os.path.expanduser("~/Masaüstü/hyz_gorev3/test")
GT_KAYNAK = os.path.expanduser(
    "~/Masaüstü/hyz_gorev3/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1_translation.csv")
KARE_DIR = os.path.join(SC, "prova_kareler")
GT_CIKTI = os.path.join(SC, "prova_gt.csv")

os.makedirs(KARE_DIR, exist_ok=True)
kaynaklar = sorted(f for f in os.listdir(KAYNAK) if f.startswith("rgb_kare"))
if not kaynaklar:
    sys.exit("kaynak kare yok")

for i in range(N):
    hedef = os.path.join(KARE_DIR, f"frame_{i:06d}.webp")
    if os.path.exists(hedef):
        continue
    img = cv2.imread(os.path.join(KAYNAK, kaynaklar[i % len(kaynaklar)]))
    cv2.imwrite(hedef, img, [cv2.IMWRITE_WEBP_QUALITY, 90])

with open(GT_KAYNAK) as f, open(GT_CIKTI, "w", newline="") as g:
    r = csv.DictReader(f)
    w = csv.writer(g)
    w.writerow(["frame_id", "x", "y", "z"])
    for i, row in enumerate(r):
        if i >= N:
            break
        w.writerow([i, row["translation_x"], row["translation_y"], row["translation_z"]])

print(f"{N} kare -> {KARE_DIR}")
print(f"GT       -> {GT_CIKTI}")
