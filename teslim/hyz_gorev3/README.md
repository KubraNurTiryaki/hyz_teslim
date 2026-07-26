# TEKNOFEST 2026 — Havacılıkta Yapay Zeka · Görev 3 (Referans Nesne Tespiti)

Bu depo, **Görev 3 (Görüntü Eşleme / Referans Nesne Tespiti)** için geliştirilen
**hibrit tespit sistemini** ve bu sistemin **resmi takım bağlantı arayüzüne**
(değerlendirme sunucusuna bağlanan arayüz) entegrasyonunu içerir.

> Amaç: Oturum başında sunucunun verdiği **referans görüntüleri** (biçerdöver, halı
> saha, kale vb.), drone kamerasından gelen karelerde bulmak ve her biri için bir
> **kutu (bbox)** üretmek. Emin olunmayan karede **hiçbir tahmin gönderilmez**
> (yanlış pozitif skoru düşürür).

---

## 📁 Depo Yapısı

| Klasör / Dosya | İçerik |
|---|---|
| `gorev3/` | Hibrit tespit sistemi (FastSAM + DINOv2 + ELoFTR). Detay: [`gorev3/README.md`](gorev3/README.md) |
| `gorev3/tools/` | Kalibrasyon (`calibrate.py`), etiketleyici (`label_minival.py`), görselleştirme (`visualize.py`) |
| `havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/` | **Sunucuya bağlanan resmi arayüz** — modelimiz `object_detection_model.py` içine bağlandı |
| `test/` | Örnek görsel tespit çıktıları (RGB + termal, kutulanmış) |
| `ham_belgeler/` | Yarışma şartnameleri |
| `THYZ_2026_Ornek_Veri_Seti/` | TEKNOFEST örnek veri seti (videolar hariç — bkz. `.gitignore`) |
| `3_Gorev_Hazir_Model_Deney_Raporu.md` | Deney ve karar raporu |
| `test_offline.py` | Model yığınının (SuperPoint/LightGlue/DINOv2/ELoFTR) çevrimdışı yüklendiğini doğrulayan hızlı test |

---

## ⚙️ Kurulum (Adım Adım)

> **Gereksinim:** 64-bit Windows/Linux, **Python 3.10+**, NVIDIA GPU (CUDA 12.4)
> önerilir. Sistem CPU'da da çalışır ama yavaştır.

### 1. Depoyu klonlayın

```bash
git clone https://github.com/KubraNurTiryaki/hyz.git
cd hyz
```

### 2. Sanal ortam oluşturun

```bash
# venv ile
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Bağımlılıkları kurun

```bash
# PyTorch (CUDA 12.4)  — CPU için: pip install torch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Model ve görüntü kütüphaneleri
pip install transformers ultralytics kornia opencv-python pillow numpy

# LightGlue + SuperPoint (kaynak repodan)
pip install git+https://github.com/cvg/LightGlue.git

# Sunucu bağlantı arayüzü bağımlılıkları
pip install python-decouple requests tqdm
```

### 4. Model ağırlıkları

Ağırlıklar **repoda yer almaz** (boyut nedeniyle), ilk çalıştırmada indirilir:

- **DINOv2** ve **ELoFTR** → HuggingFace önbelleğine otomatik iner
  (`facebook/dinov2-with-registers-base`, `zju-community/matchanything_eloftr`).
- **FastSAM** (`gorev3/weights/FastSAM-s.pt`, ~23 MB) → ilk çalıştırmada otomatik
  iner, veya `ultralytics` ile `FastSAM("FastSAM-s.pt")` çağrısıyla alınır.

### 5. (Opsiyonel) Kurulumu doğrulayın

Tüm modellerin doğru yüklendiğini kontrol edin:

```bash
python test_offline.py
```

Üç modelin de `✅` işaretiyle yüklenmesi gerekir. Ardından örnek veri üzerinde uçtan
uca test:

```bash
# Windows
set HF_HUB_OFFLINE=1
python -m gorev3.offline_test
```

---

## 🛰️ Sunucuya Bağlanma (Adım Adım)

Bu bölüm, TEKNOFEST **değerlendirme sunucusuna** bağlanmak isteyen herkes için
hazırlanmıştır. Arayüz; sunucuya giriş yapar, kareleri tek tek indirir, modelinizin
tahminini gönderir ve bağlantı koparsa **kaldığı yerden** devam eder. Sırayla takip
edin:

### Adım 1 — Arayüz klasörüne girin

```bash
cd havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
```

### Adım 2 — Giriş bilgileri dosyasını (`.env`) oluşturun

`config/example.env` dosyasını kopyalayıp adını **`.env`** yapın.
Son yol tam olarak şu olmalı: **`config/.env`**

```bash
# Windows
copy config\example.env config\.env
# Linux/Mac
cp config/example.env config/.env
```

### Adım 3 — `.env` içini kendi bilgilerinizle doldurun

`config/.env` dosyasını bir metin editörüyle açın ve TEKNOFEST'in size verdiği
bilgileri yazın:

```env
TEAM_NAME=takim_kullanici_adiniz
PASSWORD=takim_sifreniz
EVALUATION_SERVER_URL="https://sunucu-adresi/"    # sunucu açıldığında paylaşılır
SESSION_NAME=oturum_ismi                          # kullanılmıyor, boş kalabilir
```

> ⚠️ **Önemli:** `EVALUATION_SERVER_URL` mutlaka **`/`** ile bitmelidir.
>
> 🔒 **Güvenlik:** `.env` dosyanızı **kimseyle paylaşmayın** ve GitHub'a yüklemeyin.
> Bu depoda `.gitignore` sayesinde `.env` zaten hariç tutulmuştur; sadece placeholder
> içeren `example.env` şablonu paylaşılır.

### Adım 4 — ⚠️ Gerçek oturumdan ÖNCE mutlaka test edin

Sunucu, **her kare için yalnızca BİR kez** tahmin kabul eder. Bir kareye tahmin
gönderdikten sonra **üzerine tekrar gönderemezsiniz** (sunucu `406` döner). Bu yüzden:

- **Modeliniz hazır olmadan gerçek (puanlanan) oturumda `python main.py` çalıştırmayın.**
- Önce **test oturumunda** bağlantınızı deneyin.

### Adım 5 — Çalıştırın

```bash
python main.py
```

Program çalışınca:
1. Sunucuya giriş yapar ve bir **token** alır (`auth/` uç noktası),
2. Aktif oturumu ve kaldığı kareyi bulur (`progress/`),
3. Referans görüntülerini indirir (`reference/`),
4. Kareleri **birer birer** çeker (`frames/`), modeli çalıştırır, tahmini gönderir
   (`prediction/`),
5. Bağlantı koparsa, tekrar `python main.py` deyince **kaldığı kareden** devam eder.

### Çalışırken oluşan klasörler

```
TAKIM_BAGLANTI_ARAYUZU/
├── config/.env              ← sizin giriş bilgileriniz (paylaşmayın)
├── _logs/
│   └── <takım>_<tarih>.log   ← tüm işlemlerin kaydı (itirazlarda kullanılır)
└── _images/
    └── <oturum_adı>/
        ├── <kare>.webp        ← indirilen kareler
        ├── references.json    ← referans kataloğu
        └── references/        ← referans görüntüleri
```

### Sık karşılaşılan durumlar

| Mesaj / Durum | Anlamı ve çözüm |
|---|---|
| `No active session found. Exiting.` | Sunucuda aktif oturum yok. Yarışmanın/oturumun başlamasını bekleyin. |
| `Could not reach the evaluation server...` | Sunucuya ulaşılamadı **veya** kullanıcı adı/şifre yanlış. `.env` ve internet bağlantınızı kontrol edin. Ayrıntı için `_logs/` klasörüne bakın. |
| `All N frames already submitted.` | Bu oturumdaki tüm kareleri zaten gönderdiniz. Yapılacak bir şey yok. |
| `Session complete or no active session.` | Tüm kareler bitti, program normal şekilde sona erdi. |
| Bağlantı koptu | Tekrar `python main.py` çalıştırın; kaldığı kareden devam eder. |
| `Aborting: current frame is not advancing` | Tahmininiz sürekli reddediliyor (genelde hatalı veri). `_logs/` içindeki hatayı inceleyin. |

> **Kareler sıralıdır:** Sunucu, siz mevcut karenin tahminini göndermeden bir sonraki
> kareyi vermez. Kareleri atlayamaz veya ileri saramazsınız.

Kendi modelinizi entegre etme (yalnızca `src/object_detection_model.py` düzenlenir) ve
sınıf kodları için: [`havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/README.MD`](havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/README.MD)

---

## 🧠 Mimari (Özet)

- **SAM+DINO (birincil):** FastSAM ile nesne segmentlenir → DINOv2 gömme → referansla
  kosinüs benzerliği → en sıkı kutu seçilir. Kompakt / yer nesnelerinde güçlü.
- **Gri referans yolu:** gri/termal-stil referanslar CLAHE-gri domainde eşlenir
  (ref04 sorununun çözümü).
- **Sıkı seçim:** en yüksek kosinüsün marjı içindeki **en sıkı** segment seçilir
  (dev-kutu yanlış-pozitifini önler).
- **Termal (çapraz-modal):** ELoFTR + MAGSAC geometrik doğrulama; aynı modalite
  yakalanır, RGB↔termal precision öncelikli reddedilir (yanlış pozitif eklemez).
- **Yanlış-pozitif koruması:** payload'da güven skoru yok → her kutu kesin pozitif
  sayılır → emin olunmayan tespit **gönderilmez**.

Kalibrasyon: elle-görsel mini-val (30 kare) + `tools/calibrate.py` (mAP@0.5). Canlı
yarışma test sunucusunda uçtan uca doğrulandı. Ayrıntı: [`gorev3/README.md`](gorev3/README.md).

---

## 🚫 Depoda Bulunmayanlar (`.gitignore`)

Boyut/gizlilik nedeniyle şunlar depoda yer almaz; ayrıca edinilmelidir:

- **Örnek videolar** (`*.MP4`, ~1.7 GB) — TEKNOFEST örnek veri seti.
- **Model ağırlıkları** — ilk çalıştırmada otomatik iner (bkz. Kurulum · Adım 4).
- **`.env`** — yarışma kimlik bilgileri; `config/example.env` şablonundan oluşturun.
- **`_logs/`, `_images/`, `_test_output/`** — çalışma zamanı çıktıları.

---

## 📚 Detaylı Dokümanlar

- [`gorev3/README.md`](gorev3/README.md) — hibrit sistemin mimarisi, kalibrasyonu ve durumu
- [`havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/README.MD`](havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/README.MD) — arayüz kullanım kılavuzu ve model entegrasyonu
- [`3_Gorev_Hazir_Model_Deney_Raporu.md`](3_Gorev_Hazir_Model_Deney_Raporu.md) — deney/karar raporu

Başarılar! 🚀
