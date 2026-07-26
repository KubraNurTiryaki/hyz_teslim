# gorev3 — Referans Nesne Tespiti (Görev 3) hibrit sistemi

TEKNOFEST 2026 Havacılıkta YZ · 3. Görev · Görüntü Eşleme.
Metrik: **mAP** (Görev 1 ile aynı yöntem → şu an **IoU eşiği 0.5** varsayılıyor; şartname 9.3
"detaylar revizyonda" diyor). Tamamen **çevrimdışı**. Donanım: **RTX 3050 Laptop 4 GB**.

## Ne yapar
Oturum başında verilen referans nesneleri (farklı açı/irtifa/modalite) drone karelerinde bulur,
her referans için **bbox** üretir. Emin olunmayan karede **hiçbir şey göndermez** (payload'da
güven skoru yok → her kutu kesin pozitif sayılır → FP doğrudan AP düşürür).

## Mimari (hibrit, referans-tipine göre yönlendirme)
```
Kare → new_frame() [termal sezimi]
  ├─ termal kare      → crossmodal tier: ELoFTR (MatchAnything)   [VARSAYILAN KAPALI]
  └─ RGB kare:
       (1) [opsiyonel] matcher scene-tier: SuperPoint+LightGlue+MAGSAC   [VARSAYILAN KAPALI]
       (2) SAM+DINO (BİRİNCİL): FastSAM segment → DINOv2 gömme → referans kosinüs → sıkı kutu
       (3) SAM yoksa: matcher adayı + DINOv2 crop doğrulaması (fallback)
  → gate: eşik + tek-kutu (referans başına en fazla 1) → bbox | None
```

### Neden SAM+DINO birincil (ampirik, örnek veride doğrulandı)
- Matcher inlier'ları bu veride zayıf (4–6) → homografi kutusu kararsız/gevşek, IoU 0.5 tutmaz.
- DINOv2 **tüm-referans** gömmesi arka-plan baskınlığından muzdarip (çim her yeri eşler).
- **FastSAM önce nesneyi segmentler** → gömme nesneye özgü olur → ayırt edici + **sıkı kutu**.
  Görsel testte biçerdöver ve halı saha doğru ve sıkı kutulandı (cos 0.51 / 0.80).

## Dosyalar
| Dosya | İşlev |
|---|---|
| `config.py` | Tüm eşikler; `# CALIBRATE` işaretli alanlar kalibrasyona tabi |
| `embedder.py` | DINOv2 (letterbox + çok-görünüm + **batch** gömme, kosinüs) |
| `localizer_samdino.py` | FastSAM segment + DINOv2 (kare başına 1 kez segment+gömme) |
| `matcher.py` | SuperPoint+LightGlue+MAGSAC → homografi kutusu |
| `crossmodal.py` | ELoFTR termal (MAGSAC geometrik-doğrulamalı, BAĞLI) |
| `reference_matcher.py` | Hibrit router; `new_frame()` + `match_in()` |
| `gate.py` | Eşik + kutu temizliği + tek-kutu FP koruması |
| `integrate.py` | Resmi `detect()` kancasına bağlantı (kare-bağlamı cache'li) |
| `tools/visualize.py` | Tespitleri `test/` klasörüne kutulayarak çizer |
| `offline_test.py` | Örnek veri üzerinde uçtan uca offline test |
| `weights/FastSAM-s.pt` | FastSAM ağırlığı (offline cache, 23 MB) |

## Resmi arayüze BAĞLANDI (uçtan uca çalışıyor ✅)
`TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` **düzenlendi**: `__init__`'te
`ReferenceObjectDetector` yükleniyor, `detect()` Görev-3 döngüsündeki placeholder gerçek
çağrıyla değiştirildi (`sys.path`'e proje kökü eklenerek `gorev3` import ediliyor).
`python-decouple` kuruldu. Smoke-test: RGB kare 8100 → 4 geçerli `ReferencePrediction`
(biçerdöver/saha/kale), termal kare 1388 → 1 (saha). Payload formatı doğru.

### Termal katman (çapraz-modal) — BAĞLI + geometrik-doğrulamalı
`crossmodal_enabled=True`. Termal karede ELoFTR (MatchAnything) çalışır ama **ham eşleşme
değil MAGSAC inlier'ı** sinyaldir: RGB-referans→termal-kare (gerçek çapraz-modal) ham
eşleşmeleri 123'e çıksa da MAGSAC'tan sonra 5-7 inlier'da kalıyor (gürültü) → eşik
`eloftr_min_inliers=15` bunları eler (FP yok). Termal↔termal (aynı-modalite, örn. ref05
saha) 30 inlier veriyor → gerçek tespit. Hız için `crossmodal_only_gray_refs=True`:
termal karede yalnızca gri/termal referanslara ELoFTR uygulanır (RGB refler atlanır).
Termal ~2.3 sn/kare (ayrı oturum; hız puanlanmıyor).

## Çalıştırma / test
```bat
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
.venv\Scripts\python.exe -m gorev3.offline_test
```
Görsel çıktılar: `_test_output/gorev3/`. Ölçülen: ~**0.8 sn/kare** (6 referans, RTX 3050),
1.6 sn bütçe altında. Şartname: hız puanlanmıyor, ama kare kapsaması recall'ı etkiler.

## Çevrimdışı gereksinimler (yarışma makinesi)
HF cache'te olmalı (şu an mevcut): `facebook/dinov2-with-registers-base`,
`zju-community/matchanything_eloftr`. FastSAM ağırlığı `weights/` altında. `ultralytics`,
`lightglue`, `transformers`, `kornia`, `torch(cu124)` kurulu. Yarışmadan önce
`HF_HUB_OFFLINE=1` ile uçtan uca dry-run yapılmalı.

## Kalibrasyon iş akışı (mAP'i asıl belirleyen)
Araçlar `gorev3/tools/` altında. Adımlar:
```bat
:: 1) Sistem önerileriyle ön-doldurulmuş GT şablonu üret (headless)
.venv\Scripts\python.exe -m gorev3.tools.label_minival --proposals --n 12 --out mini_val.json
:: 2) mini_val.json'u aç; her kaydın present(true/false) ve bbox değerini GERÇEĞE göre düzelt
::    (GUI ile düzeltmek istersen — yerelde çalıştır:)
::    ! .venv\Scripts\python.exe -m gorev3.tools.label_minival --gt mini_val.json
:: 3) Eşikleri tara, mAP@0.5/P/R/F1 hesapla, en iyi samdino_min_cos'u öner + calibrated.json yaz
.venv\Scripts\python.exe -m gorev3.tools.calibrate --gt mini_val.json
```
`calibrated.json` `gorev3/` altına yazılırsa `config.py` onu **otomatik okur** (eşikler oradan
gelir). Matematik `--selftest` ile doğrulanmıştır. **UYARI:** `--proposals` çıktısındaki
`present` değerleri sistemin tahminidir; anlamlı kalibrasyon için **elle düzeltilmeleri şart**
(düzeltilmezse sonuç dairesel olur).

## Kalibrasyon durumu (YAPILDI — 30 kare, gri yol dahil)
Elle-görsel etiketli mini-val (`gorev3/mini_val_rgb.json`, **30 RGB kare**; tam-video
görselleştirmesiyle bulunan 6 negatif kare eklendi). İki ayrı eşik (`gorev3/calibrated.json`,
`config.py` otomatik okur):
- **Renkli referanslar** (01/02/03/05/06): `samdino_min_cos = 0.55` → AP@0.5=0.990, **F1=0.962**.
- **Gri referans** (04, CLAHE-gri yol): `samdino_min_cos_gray = 0.48` → **F1=0.75, P=1.00** (FP=0).

### Görselleştirme + FP temizliği (`tools/visualize.py`)
Kalibre sistem tüm videoda çalıştırılıp bulunan nesneler `test/` klasörüne kutulanarak çizilir
(her kutu + eşleşen referans thumbnail'i + tier/cos). Bu, iki üretim-FP'sini ortaya çıkardı ve
düzeltildi: (1) FastSAM'in **tüm-kareyi-kaplayan arka plan segmenti** dev gri FP kutular
üretiyordu → `seg_max_area_frac=0.6` ile segmentasyon aşamasında elendi; (2) gri eşik 0.33
üretimde çok düşüktü → negatif kareler val'e eklenip 0.48'e yükseltildi (gri FP=0). Kalan tek FP:
renkli ref01'in bir kamyonu biçerdövere eşlemesi (0.61) — kaldırmak gerçek biçerdöverleri
kaybettirdiği için kabul edildi.

### ref04 (gri referans) sorunu — CLAHE ile ÇÖZÜLDÜ
Kök neden (deneyle bulundu): **rotasyon augmentasyonu** gri biçerdöver görünümlerini beyaz binalara
eşliyordu (rgb çok-görünüm gap −0.11). Çözüm (`embedder.py`/`localizer_samdino.py`/`reference_matcher.py`):
gri referanslar (`is_grayscale_image` ile otomatik tespit) ve aday segmentler **CLAHE-normalize gri
domain + tek-görünüm** (rotasyonsuz) embed edilir, ayrı `samdino_min_cos_gray` eşiği kullanılır.
Sonuç: ref04 katkısı **F1≈0.5 → 0.8**; beyaz-bina FP'si öldü (0.556→0.259), 2 biçerdöver kaçışı
kurtarıldı. Maliyet: gri referans aktifken segmentler ikinci kez (gri) embed edilir → ~0.8→1.6 sn/kare
(hız puanlanmıyor; 2250 kareyi kapsar). Kapatmak için `config.gray_ref_clahe=False`.

## KALAN İŞ — kritik önem sırası
1. **Çevrimdışı uçtan-uca dry-run:** yarışma makinesinde `main.py` + sunucu simülasyonu ile tam
   akış (`HF_HUB_OFFLINE=1`) test edilmeli. Tekil `detect()` yolu smoke-test'ten geçti.
2. **Kalibrasyonu genişlet:** yer-seviyesi kale (R05), kabin (R06) ve daha çok termal kareyle
   örnekle. Alan sınırları ve `eloftr_min_inliers` de ayarlanır.
3. **ref04 artık kaçış (6600) + kamyon FP (8880):** gri domainde bazı biçerdöver açıları düşük
   skorlu; renkli ref01 bir kamyonu 0.61 ile biçerdövere eşliyor. Referans-başına eşik / şekil
   doğrulaması denenebilir.
4. **Kare kapsaması / hız:** budget aşılırsa `fastsam_imgsz` düşür (768→512) veya kareleri
   zamansal örnekle.
4. **Çoklu örnek / tam-kopya:** tek-kutu kuralı şu an referans başına 1; aynı referanstan
   birden çok örnek varsa gate genişletilmeli.
