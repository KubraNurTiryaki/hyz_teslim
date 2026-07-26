# HYZ YEDEK — 26 Temmuz 2026

Bilgisayar temizlenip repodaki MASTER PROMPT ile sıfırdan kurulmadan önce alınan
yedek. Buradaki her şey GitHub'da YOK — kaybedilirse geri gelmez.

## teslim/
T3 KYS'ye yüklenecek kaynak kod teslimi (378 dosya). Üç deponun süzülmüş hali:
ana repo + SP_SLAM3 (Görev 2 C++) + hyz_gorev3 (Görev 3).
Rapor, çıktı, veri seti, derlenmiş binary İÇERMEZ. Görev 1 modeli dahildir.
**Son tarih: 30 Temmuz 2026 Perşembe 17:00 TSİ.**
Yükleyen: takım danışmanı veya kaptan (üyeler yükleyemez).

## video_metinleri/
Jüriye gönderilecek 5 dakikalık videonun konuşma metinleri.
- VIDEO_KONUSMA_METNI_2_JURI.md  ← TESLİM EDİLECEK OLAN (mail şartlarına göre)
- VIDEO_KONUSMA_METNI.md         ← ilk taslak (alternatif)

## commitlenmemis_isler/
Depolarda olmayan, bu makinede yazılmış kodlar:
- BU_MAKINEDE_CALISTIRMA.md — bu makinedeki komut rehberi
- gorev2_duman_testi.py      — SLAM duman testi (veri gerekmez)
- prova_verisi_uret.py       — yerel prova verisi üreteci
- gorev3_kare_demo.py        — Görev 3 kare demosu

## yamalar/
Temiz klonlara uygulanacak düzeltmeler:
- ana_repo_yol_duzeltmeleri.patch      — /home/zeylo sabit yollarını taşınabilir yapar (11 dosya)
- gorev3_test_offline_duzeltmesi.patch — SuperPoint tensör boyutu düzeltmesi

Uygulama:
    cd <yeni_klon> && git apply /home/kurt/hyz_YEDEK_2026-07-26/yamalar/<dosya>.patch
