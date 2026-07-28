# TEKNOFEST 2026 — Havacılıkta Yapay Zeka · Takım hamidiye_4907501

Üç görevi **tek istemcide, tek süreçte** birleştiren yarışma sistemi.
16 Temmuz 2026 çevrim içi oturumunda **2250/2250 kareyi 53,5 dakikada** işleyip
gönderdi (0 ret, 0 kopma): 6834 nesne kutusu (Görev 1), kesintideki 1547 karenin
tamamında SLAM konumu (Görev 2), 6 referans nesnede 304 pencere-içi kutu (Görev 3).

| | |
|---|---|
| **Görev 1** | Nesne tespiti — YOLO26-large, 4 sınıf, üç kademeli hat |
| **Görev 2** | GPS'siz konum — SP-SLAM3 (SuperPoint + LightGlue) + yansımalı planar hizalama |
| **Görev 3** | Referans nesne — FastSAM + DINOv2 (+ termalde ELoFTR/MAGSAC) |

---

# 1) Kurulum adımları

Hedef sistem: **Ubuntu 22.04**, NVIDIA GPU (≥6 GB VRAM), ~40 GB boş disk.
Ayrıntılı ve doğrulama komutlu sürüm: **`KURULUM.md`** (10 bölüm, her bölümün
sonunda doğrulama adımı vardır). Aşağıdaki özet o belgeyle birebir uyumludur.

### 1.1 Sistem paketleri
```bash
sudo apt update && sudo apt install -y build-essential cmake git pkg-config \
  libopencv-dev libeigen3-dev libboost-serialization-dev libssl-dev \
  libgl1-mesa-dev libglew-dev libpython3-dev python3-venv python3-pip \
  python3-tk ffmpeg git-lfs
git lfs install
```

### 1.2 Depolar — TAM bu yollara
Kodlardaki varsayılanlar bu yolları bekler (hepsi ortam değişkeniyle
değiştirilebilir; bkz. §4.4).
```bash
git clone https://github.com/kayranecatikara/thyz2026-hamidiye.git ~/Masaüstü/teknofest_gorev2
git clone https://github.com/kayranecatikara/SP_SLAM3.git          ~/SP_SLAM3
git clone https://github.com/KubraNurTiryaki/hyz.git               ~/Masaüstü/hyz_gorev3

mkdir -p ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi
cp -r ~/Masaüstü/teknofest_gorev2/istemci/TAKIM_BAGLANTI_ARAYUZU \
      ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/
```
LFS kontrolü: `du -h ~/SP_SLAM3/lightglue.pt` → **~46 MB** olmalı.
Birkaç KB ise `cd ~/SP_SLAM3 && git lfs pull`.

> **Bu ZIP'i kullanıyorsanız:** depoları klonlamak yerine ZIP içeriğini
> `~/Masaüstü/teknofest_gorev2` olarak açabilirsiniz; `SP_SLAM3/` ve
> `hyz_gorev3/` alt klasörleri ZIP'in içindedir.

### 1.3 Pangolin v0.8 (SLAM görselleştirme)
```bash
git clone https://github.com/stevenlovegrove/Pangolin.git ~/Pangolin
cd ~/Pangolin && git checkout v0.8 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install && sudo ldconfig
```

### 1.4 LibTorch 2.3.0 (cu121, C++)
```bash
cd /tmp && wget "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip"
unzip -q libtorch-*.zip && sudo mv libtorch /usr/local/libtorch
echo "/usr/local/libtorch/lib" | sudo tee /etc/ld.so.conf.d/libtorch.conf && sudo ldconfig
```
cuDNN ayrıca gerekmez; paket kendi kopyasını taşır.

### 1.5 SP_SLAM3 derlemesi + sözlük
```bash
cd ~/SP_SLAM3 && ./build.sh
./tools/convert_vocab Vocabulary/superpoint_voc.yml.gz Vocabulary/superpoint_voc.dbow3
ls Examples/Monocular/mono_folder_watch && echo "SLAM HAZIR"
```
> **Bellek uyarısı:** ana derleme adımında `cc1plus` çeviri birimi başına ~2,9 GB'a
> çıkabilir. 16 GB'ın altındaki makinelerde geçici takas açıp düşük paralellik
> kullanın:
> ```bash
> sudo fallocate -l 8G /swapfile-derleme && sudo chmod 600 /swapfile-derleme
> sudo mkswap /swapfile-derleme && sudo swapon /swapfile-derleme
> cd ~/SP_SLAM3/build && make -j4
> ```
> Sözlük dönüşümü ~7 dakika sürer.

### 1.6 İki Python ortamı
**(a) Sistem `python3`** — yarışma istemcisi bununla koşar:
```bash
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install --user ultralytics transformers kornia opencv-python pillow numpy \
                    python-decouple requests tqdm
pip3 install --user "git+https://github.com/cvg/LightGlue.git"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"   # True olmalı
```
**(b) `~/venvs/slam`** — analiz/panel araçları:
```bash
python3 -m venv ~/venvs/slam
~/venvs/slam/bin/pip install numpy opencv-python matplotlib
```

### 1.7 Model dosyaları
Bkz. **§3** ve `model.txt`.

### 1.8 Kimlik bilgileri (yalnız canlı sunucuya bağlanacaksanız)
```bash
cd ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
cp config/example.env config/.env && nano config/.env
```
`TEAM_NAME` / `PASSWORD` / `EVALUATION_SERVER_URL` (sonu `/` ile biter).
Bu dosya teslim paketinde **yoktur**, yalnızca `example.env` şablonu vardır.

### 1.9 Kurulum doğrulaması
```bash
cd ~/Masaüstü/teknofest_gorev2
~/venvs/slam/bin/python alignment.py     # "alignment.py oz-testleri tamam."
python3 -c "
import sys; sys.path.insert(0,'gorev_1')
from yarisma_pipeline import model_yukle
assert model_yukle('gorev_1/birincil_run7_26l.pt').names == {0:'tasit',1:'insan',2:'uap',3:'uai'}
print('G1 OK')"
python3 gorev2_duman_testi.py            # SLAM duman testi, veri gerekmez
```

---

# 2) Gerekli bağımlılıklar ve sürüm bilgileri

Aşağıdaki sürümler sistemin **doğrulandığı** ortamdan alınmıştır.

### Sistem
| Bileşen | Sürüm |
|---|---|
| İşletim sistemi | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| GCC | 11.4.0 |
| CMake | 3.22.1 |
| NVIDIA CUDA Toolkit | 12.2 (V12.2.140) |
| GPU (doğrulandığı) | NVIDIA RTX 3060, 8 GB |

### C++ tarafı (SP-SLAM3)
| Bileşen | Sürüm |
|---|---|
| LibTorch | 2.3.0+cu121 (cxx11-abi, shared-with-deps) |
| Pangolin | v0.8 |
| OpenCV | 4.5.4 (`libopencv-dev`, apt) |
| Eigen | 3.4.0 |
| Boost (serialization) | 1.74 |
| DBoW3, g2o, Sophus | `SP_SLAM3/Thirdparty/` içinde, kaynaktan derlenir |

### Python — sistem `python3` (yarışma istemcisi)
| Paket | Sürüm |
|---|---|
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| ultralytics | 8.4.105 |
| transformers | 5.14.1 |
| kornia | 0.8.2 |
| opencv-python | 4.11.0 |
| pillow | 12.3.0 |
| numpy | 1.26.4 |
| requests | 2.25.1 |
| tqdm | 4.69.1 |
| python-decouple | istemci `requirements.txt` |
| lightglue | `git+https://github.com/cvg/LightGlue.git` |

> **Not:** torch `cu130` sürümüyle doğrulanmıştır (makinenin NVIDIA sürücüsü
> CUDA 13.0 sunuyor). `cu121` tekerleği ile de çalışır. C++ tarafı bundan
> bağımsızdır: LibTorch 2.3.0+cu121 ve CUDA Toolkit 12.2 kullanır. İkisi ayrı
> süreç olduğu için çakışmaz.

### Python — `~/venvs/slam` (analiz/panel)
| Paket | Sürüm |
|---|---|
| numpy | 2.2.6 |
| opencv-python | 5.0.0 |
| matplotlib | 3.10.9 |

---

# 3) Model dosyaları

Üç görevin modelleri **toplam 727 MB**'dir; şartnamedeki 500 MB sınırını aştığı
için **Google Drive** üzerinden paylaşılmıştır. Bağlantı ve tam liste bu klasörün
kökündeki **`model.txt`** dosyasındadır.

| Görev | Modeller | Boyut |
|---|---|---|
| 1 | `birincil_run7_26l.pt` (YOLO26-large) | 51 MB |
| 2 | `superpoint.pt`, `lightglue.pt`, `cosplace.pt`, `superpoint_voc.yml.gz` | 212 MB |
| 3 | `FastSAM-s.pt`, DINOv2, ELoFTR, SuperPoint/LightGlue `.pth` | 465 MB |

Görev 1 modeli **bu ZIP'in içinde de** bulunur (`gorev_1/birincil_run7_26l.pt`),
böylece Görev 1 Drive'a gerek kalmadan çalıştırılabilir.

Görev 3 modelleri internet varken **ilk çalıştırmada otomatik iner**; Görev 2
modelleri SP_SLAM3 deposunda git-lfs ile gelir. Drive paketi çevrimdışı kurulum
ve bütünlük doğrulaması içindir (`SHA256SUMS.txt`).

---

# 4) Çalıştırma komutları

### 4.1 Yarışma koşusu (asıl komut)
```bash
cd ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
HF_HUB_OFFLINE=1 python3 main.py
```
Sunucuya girer → kaldığı kareyi bulur → referansları indirir → kareleri tek tek
çekip üç görevi de çalıştırır → tahminleri gönderir → oturum bitince
`Session complete` der ve çıkar. Başka hiçbir şey çalıştırmak gerekmez.
Ayrıntı ve arıza tablosu: **`YARISMA_GUNU.md`**.

### 4.2 Üç görevi örnek veri seti üzerinde çalıştırma (demo)
```bash
cd <bu_klasor>
python3 demo_ornek_veri.py
```
Yarışma istemcisinin `detect()` fonksiyonunun **aynısını** çağırır; tek fark
kareleri sunucudan değil resmî örnek videodan okumasıdır. Her kare için tek satır
basar:
```
kare  537 KESINTI| G1:3 tasitx2 insanx1  | G2:xyz=( -32.1, 84.0, 10.4) hata= 0.7m | G3:4 Ref01 Ref02 Ref03 Ref04
```
Seçenekler: `--bas`, `--kalibrasyon`, `--kor`, `--g3-pencere`.
Örnek veri setinin yolu betiğin başındaki `VERI` değişkenindedir.

### 4.3 Uçtan uca yerel prova (resmî protokol taklidi)
```bash
# Terminal 1 — resmî sunucunun yerel kopyası
cd <bu_klasor> && python3 resmi_mock.py --limit 600 --drop 450-599

# Terminal 2 — yarışma komutunun TA KENDİSİ, sadece adres yerelde
cd ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
EVALUATION_SERVER_URL="http://127.0.0.1:5580/" HF_HUB_OFFLINE=1 python3 main.py
```
Son satır `Session complete` ise tesisat sağlamdır.

### 4.4 Ortam değişkenleri (hepsi opsiyonel)
| Değişken | Varsayılan | İşlevi |
|---|---|---|
| `GOREV2_DIR` | `~/Masaüstü/teknofest_gorev2` | Motor ve Görev 1/2 kökü |
| `GOREV1_DIR` | `$GOREV2_DIR/gorev_1` | Görev 1 hattı + model |
| `GOREV3_DIR` | `~/Masaüstü/hyz_gorev3` | Görev 3 paketi |
| `GOREV2_SETTINGS` | otomatik seçilir | SLAM kalibrasyon yaml'ını elle dayatır |
| `GOREV2_RUN_DIR` | `$GOREV2_DIR/run_yarisma` | SLAM çalışma klasörü |
| `HF_HUB_OFFLINE` | — | `1` ise Görev 3 modelleri yalnız önbellekten okunur |

### 4.5 Analiz ve doğrulama araçları
```bash
~/venvs/slam/bin/python alignment.py                           # hizalama öz-testleri
python3 gorev2_duman_testi.py                                  # SLAM duman testi
python3 evaluate_denklem2.py --pred <pred.csv> --gt <gt.csv>   # yarışma metriği
~/venvs/slam/bin/python analiz_3eksen.py --gt <gt.csv> --pred <pred.csv> \
    --kaydet analiz_3eksen/cikti.png                           # x/y/z gerçek-vs-tahmin
./video_analiz.sh <1-6>                                        # canlı Pangolin + panel
```

---

# 5) Kod akışı ve açıklamalar

## 5.1 Genel mimari

```
python3 main.py   (resmî TAKIM_BAGLANTI_ARAYUZU — yalnız object_detection_model.py değiştirildi)
        │  kareyi indirir → detect() → tahmini gönderir → sonraki kare
        ▼
┌─ Görev 1 ──────────┐ ┌─ Görev 2 ────────────────┐ ┌─ Görev 3 ─────────────┐
│ YOLO26l + ped      │ │ SP-SLAM3 (C++ süreç)     │ │ FastSAM + DINOv2      │
│ hakemi + ped-içi   │ │ SuperPoint + LightGlue   │ │ + ELoFTR (termal)     │
│ insan + ego-hareket│ │ + yansımalı planar       │ │ emin değilse kutu     │
│ takibi             │ │   hizalama               │ │ GÖNDERMEZ             │
└─ gorev_1/          ┘ └─ gorev2_engine.py+SP_SLAM3┘ └─ hyz_gorev3/         ┘
```

**İki süreç, bir köprü.** Python süreci Görev 1 ve Görev 3'ü aynı GPU'da
çalıştırır. SLAM ayrı bir **C++ sürecidir** (`mono_folder_watch`);
`gorev2_engine.py` onu `subprocess.Popen` ile başlatır (satır 149-159).
Haberleşme dosya sistemi üzerindendir: Python kareyi `run_*/inbox/`'a yazar,
C++ bulduğu pozu `run_*/outbox/`'a yazar, Python o dosyayı izler. Bu tasarım,
C++ tarafı çökse bile Python döngüsünün ayakta kalmasını sağlar.

**Çelik zırh ilkesi.** Sunucu, tahmin göndermeden bir sonraki kareyi vermez;
tek karede takılmak oturumu kaybettirir. Bu yüzden her görev kendi
`try/except` bloğundadır ve **her kareye tam 1 geçerli tahmin** gider:
Görev 1 çökerse boş tespit, Görev 2 çökerse ölü-hesap konumu.

## 5.2 Bir karenin yolculuğu — `detect()`

`istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` **satır 181-297**

| Adım | Ne olur |
|---|---|
| 1 | Kare diskten **bir kez** okunur (`cv2.imread`); aynı görüntü Görev 1 ve 2'ye verilir |
| 2 | **Görev 1** → `g1.kare_isle(kare)` → `DetectedObject` listesi |
| 3 | **Görev 2** → `engine.process_frame(fid, kare, ref_xyz, health)` → `DetectedTranslation` |
| 4 | **Görev 3** → pencere içindeki her referans için `detect_for_frame()` → `ReferencePrediction` |
| 5 | Üçü de aynı `prediction` nesnesine yazar; `create_payload()` sunucu JSON'unu üretir |

### Girdi / çıktı formatları

| Görev | Girdi | Çıktı (sunucuya giden alanlar) |
|---|---|---|
| 1 | BGR kare (4K / 1080p) | `cls` (URL), `landing_status`, `moving_status`, `top_left_x/y`, `bottom_right_x/y` |
| 2 | BGR kare + `health_status` (`"0"`/`"1"`) + sağlıklıysa gerçek `x,y,z` | `translation_x`, `translation_y`, `translation_z` |
| 3 | BGR kare + referans görüntü yolu + pencere (`frame_start/end_image_url`) | `reference` (URL), `top_left_x/y`, `bottom_right_x/y` |

Kodlar `src/constants.py`'dedir: sınıf `Tasit=0, Insan=1, UAP=2, UAI=3`;
iniş `Inilebilir=1 / Inilemez=0 / Inis Alani Degil=-1`;
hareket `Hareketli=1 / Sabit=0 / Tasit Degil=-1`.

## 5.3 Görev 1 — `gorev_1/yarisma_pipeline.py`

`kare_isle()` (**satır 146-190**) üç kademeli bir hattır:

1. **Tam kare taraması** — `_det(kare, 1280, ...)` (satır 149). Taşıt/insan doğrudan
   kabul edilir; UAP/UAİ adayları kuyruğa alınır.
2. **Ped hakemi** — `_det(crop, 192, 0.10)` (satır 164). Aday ped kırpılıp modele
   yeniden sorulur; 4K karede minicik görünen ped, kırpılıp büyütülünce çok daha
   net ayırt edilir. Eşiği geçemeyen aday elenir.
3. **Ped-içi insan taraması** — `_det(crop, 640, 0.10)` (satır 168). Onaylanan
   pedin içinde insan varsa `inis=0` (inilemez) yazılır.

Ardından **ego-hareket telafisi**: `_ego.guncelle()` ORB + RANSAC ile kameranın
kendi hareketini kestirip çıkarır; geriye kalan gerçek nesne hareketidir. Böylece
uçan kameradan "hareketli mi sabit mi" doğru ayırt edilir.

Ölçülen hız: 4K karede p95 = **70 ms**. Model mAP@0.5 = **0.849**.

## 5.4 Görev 2 — `gorev2_engine.py` + `alignment.py`

`process_frame()` (**satır 297-340**) sağlık bitine göre dallanır:

- **`health=1`** → gerçek konum zaten elde; **aynen geri gönderilir** (hata 0).
  Kare yine de SLAM'e beslenir ve `(SLAM pozu, gerçek konum)` çifti biriktirilir.
  Kaynak: `echo`.
- **`health=0`** (kör bölge) → biriken çiftlerden öğrenilen dönüşümle SLAM pozu
  metreye çevrilir. Kaynak: `slam`. SLAM izlemeyi kaybederse ölü-hesap devreye
  girer (kaynak `deadreckon`) — kare asla boş gitmez.

### Projedeki en kritik bulgu — yansıma (el-yönü) sorunu

`alignment.py` → `_umeyama_2d()` (**satır 112-121**).

SLAM çerçevesi ile GT çerçevesi arasında **yansıma** vardır. Klasik Umeyama
`det=+1` kısıtlıdır, yansımayı temsil edemez; varyansı büyük olan xy'yi eşleyip
**z'yi feda eder** (ters çevirir). Çözüm: hizalamayı her zaman **yansımalı 2B (xy)
Procrustes + ayrı 1B (z) ölçek/öteleme** olarak çözmek:

```python
if allow_reflection:
    S_refl = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) >= 0:
        S_refl[1, 1] = -1.0          # ← yansıma adayı
    cand = build(S_refl)
    if cand[3] < best[3]:            # artığı küçükse onu seç
        best = cand
```

Etkisi (aynı oturum, aynı SLAM çıktısı, yalnız hizalama matematiği farklı):
**33,8 m → 6,2 m**; z ekseni hatası **27,7 m → 3,2 m**.
Görseller: `gorseller/01_hizalama_BOZUK_33.8m.png`, `02_hizalama_DUZELTILMIS_6.2m.png`.

### Diğer kritik kararlar
- **Init nokta tavanı** (`Tracking.cc`): LightGlue ~1600 noktanın üstünde bozuluyor;
  init çıkarıcısı ≤1200'e sınırlandı → "init'te takılma" tamamen bitti.
- **Kamera otomatiği**: ilk karenin genişliğinden kalibrasyon seçilir
  (4000→4K, 3840→cropA, 1920→1080p, 640→termal).
- **`blend_tau=0`**: SLAM→ölü-hesap harmanı sim3 döneminin yamasıydı; planar
  hizalama driftin kökünü çözdükten sonra ölçüldü ve **kötüleştirdiği** görüldü
  (11-15 m → 44-73 m), varsayılan olarak kapatıldı. Acil geri dönüş bayrağı
  olarak kodda durur.
- **Sıkıştırma**: kareler SLAM'e yüksek kalitede verilir; agresif JPEG sıkıştırma
  SuperPoint'i zayıflatıp izlemeyi kaybettiriyor.

Doğrulanan sonuçlar: 2025 oturumlarında yarışma metriğiyle **4,3 – 8,8 m**.
2026 örnek videosunda toplam 33,6 m; bunun 32 m'si z ekseninden gelir
(x 4,0 / y 2,2 m). Sebebi: kalibrasyon dakikasında uçuş neredeyse düz gidiyor;
tek kameralı SLAM'de yükseklik ölçeği ancak irtifa değişimi gözlemlenirse
çözülebilir. Bu bir kod hatası değil, tek kameranın fiziksel sınırıdır.

## 5.5 Görev 3 — `hyz_gorev3/gorev3/`

Burada bir **sınıf** aranmaz; verilen tek örnek fotoğraftaki **o nesne** bulunur.

| Aşama | Dosya | Ne yapar |
|---|---|---|
| Segmentasyon | `localizer_samdino.py` | FastSAM kareyi bölgelere ayırır; tüm kareyi kaplayan arka plan segmenti aday dışı bırakılır |
| Gömme | `dino_embed.py` | Her segmentin DINOv2 vektörü; referans çok ölçekli gömülür |
| Eşleme | `reference_matcher.py` | Kosinüs benzerliği + tier yönlendirmesi (`compact` / `gray` / `crossmodal`) |
| Termal yol | `crossmodal.py` | ELoFTR + MAGSAC; **renkli** referansı **termal** karede arar |
| Entegrasyon | `integrate.py` | `detect_for_frame()`; oturum adında "termal" geçerse çapraz-modal yolu açar |

**Yanlış-pozitif koruması** (`object_detection_model.py` **satır 292**):
```python
if not bbox:
    continue          # emin değilsek HİÇBİR kutu gönderme
```
Arayüzde güven skoru alanı yoktur; gönderilen her kutu kesin iddia sayılır ve
yanlış pozitif ceza getirir. Bu yüzden sistem emin olmadığı karede sessiz kalır.
Referans nesne kadrajda yoksa hiç kutu üretmemesi **beklenen** davranıştır.

## 5.6 Depo haritası

| Yol | İçerik |
|---|---|
| `istemci/TAKIM_BAGLANTI_ARAYUZU/` | Resmî yarışma istemcisi; **yalnız `src/object_detection_model.py` değiştirildi** |
| `gorev2_engine.py`, `alignment.py` | Görev 2 motoru + hizalama matematiği |
| `gorev_1/` | Görev 1 hattı + `birincil_run7_26l.pt` |
| `hyz_gorev3/` | Görev 3 hibrit sistemi |
| `SP_SLAM3/` | SLAM C++ kaynağı (modeller Drive'da / git-lfs'te) |
| `demo_ornek_veri.py` | Üç görevi örnek veri setinde çalıştıran demo |
| `resmi_mock.py`, `mock_server.py`, `bridge.py` | Prova altyapısı |
| `tam_prova.py`, `analiz_3eksen.py`, `evaluate_denklem2.py` | Uçtan uca prova ve ölçüm |
| `gorseller/` | Sonuç grafikleri, kutulu kareler, canlı koşu klibi |
| `KURULUM.md` | Sıfırdan kurulum, 10 bölüm, doğrulamalı |
| `YARISMA_GUNU.md` | Yarışma günü tek komut + arıza tablosu |
| `SETUP_LOG.md` | Tüm geliştirme/karar günlüğü, gerekçeleriyle |
| `VIDEO_KONUSMA_METNI.md` | Teslim videosunun çekim listesi ve metni |
| `model.txt` | Model dosyalarının Google Drive bağlantısı ve listesi |

---

# 6) Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `lightglue.pt` birkaç KB | `cd ~/SP_SLAM3 && git lfs pull` |
| Derlemede `cc1plus` öldürüldü | Takas aç + `make -j4` (bkz. §1.5) |
| SLAM cmake'te CUDA bulunamadı | CUDA Toolkit 12.2 kurulu değil |
| İlk karede ~30 sn duraklama | Normal — SLAM sözlüğü yükleniyor |
| `CUDA out of memory` | 8 GB kartta üç görev sınırda çalışır; `nvidia-smi` ile artık süreç var mı bak, öldür, yeniden başlat |
| Görev 3 hiç kutu göndermiyor | Referans nesne kadrajda yoksa beklenen davranış (FP koruması) |
| `Could not import AutoImageProcessor` | `pip3 install --user "pillow>=10.4"` |

---

*Sistem 15-16 Temmuz 2026'da geliştirildi ve canlı yarışmada doğrulandı.
Kararların tam gerekçeleri için `SETUP_LOG.md`.*
