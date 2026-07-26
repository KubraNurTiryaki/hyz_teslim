# BU MAKİNEDE ÇALIŞTIRMA — komut rehberi

Kurulum 25 Temmuz 2026'da bu makinede (Ubuntu 22.04, RTX 3060 8 GB, kullanıcı `kurt`)
yapıldı ve doğrulandı. Sıfırdan kurulum için `KURULUM.md`, yarışma günü akışı için
`YARISMA_GUNU.md`. Bu dosya **hangi komutu ne için çalıştıracağını** anlatır.

## Yollar (bu makinede)

| Ne | Yol |
|---|---|
| Bu depo | `~/Masaüstü/hyz_video_teslim_icin` |
| Kodların beklediği ad | `~/Masaüstü/teknofest_gorev2` → **bu depoya symlink** |
| SLAM (C++) | `~/SP_SLAM3` (derlendi) |
| Görev 3 deposu | `~/Masaüstü/hyz_gorev3` |
| Yarışma istemcisi | `~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU` |
| Panel/prova venv'i | `~/venvs/slam/bin/python` |
| İstemci + Görev 1/3 | sistem `python3` (torch cu130, CUDA ✓) |

---

## 0) Yarışma komutu (üç görev birlikte — asıl komut)

```bash
cd ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
HF_HUB_OFFLINE=1 python3 main.py
```

Öncesinde `config/.env` doldurulmalı (şu an şablon):
```bash
nano ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU/config/.env
# TEAM_NAME / PASSWORD / EVALUATION_SERVER_URL (sonu "/" ile bitecek)
```

---

## 1) Görev 1 — Nesne Tespiti (tek başına)

**Bir video işle, kutulu çıktı videosu üret:**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin/gorev_1
python3 video_analiz.py /yol/video.mp4      # -> analiz_cikti.mp4
```

**Sadece hız ölçümü (kutu çizmeden):**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin/gorev_1
python3 yarisma_pipeline.py /yol/video.mp4
```

**Model + sınıf sırası doğrulaması:**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
python3 -c "
import sys; sys.path.insert(0,'gorev_1')
from yarisma_pipeline import model_yukle
print(model_yukle('gorev_1/birincil_run7_26l.pt').names)"
# {0:'tasit',1:'insan',2:'uap',3:'uai'} çıkmalı
```

---

## 2) Görev 2 — GPS'siz Konum Kestirimi

**Hizalama matematiği öz-testleri (veri gerekmez, 1 sn):**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
~/venvs/slam/bin/python alignment.py
```

**SLAM motoru duman testi (veri seti gerekmez, ~1 dk):**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
python3 gorev2_duman_testi.py
```
SLAM sürecinin ayağa kalktığını, sağlık=1'de referansın aynen döndüğünü (hata 0)
ve sağlık=0'da NaN'sız konum üretildiğini doğrular.
Bu makinede sonuç: `echo: 6, deadreckon: 6` — hepsi geçerli, yankı hatası 0.

**Tam prova (2025 oturum kareleri gerekir — `KURULUM.md` §9):**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
./canli_prova.sh 2                 # Pangolin görüntüsü + canlı panel
./video_analiz.sh 2                # penceresiz toplu koşu
```

**Metrik (Denklem-2) hesabı:**
```bash
python3 evaluate_denklem2.py --pred prova2025/pred_analiz_o2.csv \
                             --gt   prova2025/oturum2_gt.csv
```

---

## 3) Görev 3 — Referans Nesne Tespiti

```bash
cd ~/Masaüstü/hyz_gorev3
python3 gorev3_kare_demo.py                # RGB kareler + 6 referans
python3 gorev3_kare_demo.py --termal       # termal kareler
python3 gorev3_kare_demo.py --kare 10      # daha çok kare
```
Çıktı: `~/Masaüstü/hyz_gorev3/_test_output/gorev3/` altında kutulu jpg'ler.

**Modellerin yüklendiğini doğrula:**
```bash
cd ~/Masaüstü/hyz_gorev3 && python3 test_offline.py
```
> Not: bu betiğin SuperPoint bölümü 3 boyutlu tensör verdiği için "❌" yazar;
> gerçek kod (`gorev3/matcher.py`) 4 boyutlu verir ve çalışır. DINOv2 ve
> MatchAnything satırlarında ✅ görmen yeterli.

---

## 4) Üç görev birlikte — YEREL prova (sunucuya dokunmaz)

Gerçek yarışma komutunun aynısı, sadece URL yerelde.

**Önce prova verisi üret** (gerçek veri seti yokken; depodaki örnek kareleri
döngüsel tekrarlar, GT'yi 2026 örnek `translation.csv`'sinden alır):
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
python3 prova_verisi_uret.py 300      # -> prova_yerel/prova_kareler + prova_gt.csv
```
> Bu bir **tesisat testidir**: kareler ardışık bir uçuş değildir, dolayısıyla SLAM
> izleme kuramaz ve Görev 2 ölü-hesaba düşer. Ölçtüğü şey protokol, üç görevin
> tek süreçte koşması, payload serileştirmesi ve gönderim döngüsüdür — konum
> doğruluğu değil. Doğruluk için gerçek oturum kareleri gerekir (§5).

**Terminal 1 — resmî protokolün yerel taklidi:**
```bash
cd ~/Masaüstü/hyz_video_teslim_icin
python3 resmi_mock.py --frames-dir prova_yerel/prova_kareler \
        --gt prova_yerel/prova_gt.csv --limit 300 --drop 200-299 \
        --port 5580 --session YEREL_PROVA
```

**Terminal 2 — yarışma komutunun TA KENDİSİ:**
```bash
cd ~/Masaüstü/test/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
rm -rf _images/YEREL_PROVA
EVALUATION_SERVER_URL="http://127.0.0.1:5580/" HF_HUB_OFFLINE=1 python3 main.py
```
Son satır `Session complete` ise tesisat sağlam. Gönderilen paketler
`mock_gonderimler.jsonl`e yazılır. Bitince Terminal 1'i Ctrl+C ile kapat.

Kendi verinle koşacaksan: `<kare_dizini>` `frame_%06d.webp` adlandırması,
`<gt.csv>` ise `frame_id,x,y,z` başlığı bekler.

### Bu makinede alınan sonuç (25 Tem 2026)
| Ölçüm | Değer |
|---|---|
| Gönderilen kare | **300/300**, 0 boş payload, `Session complete` |
| Görev 1 | 623 kutu (2.1/kare) |
| Görev 2 | 300/300 karede konum; kesintide 100 kare ölü-hesap (fail-safe) |
| Görev 3 | 2 pencerede 102 referans kutusu |
| Kamera seçimi | `genislik=1920 -> teknofest_1080p.yaml` (otomatik) |
| Log'da `YUKLENEMEDI`/`ERROR` | 0 |

---

## 5) Bu makinede EKSİK olan tek şey: veri setleri

2025 oturum kareleri ve 2026 örnek videoları depoda yok (GB'larca; `.gitignore`).
Kaynaklar `KURULUM.md` §9'da. İndirdikten sonra:
- 2025 kareleri → `prova2025/frames_o2`, `frames_o3`, `frames_o4` …
- 2026 videosu → kare dizisine çevirme komutu `KURULUM.md` §9'da.

Veri gelmeden çalışanlar: Görev 1 (herhangi bir video/kare ile), Görev 3 (depodaki
örnek kareler), alignment öz-testleri, Görev 2 duman testi, yerel prova (tesisat).

---

## 6) Sık sorunlar

| Belirti | Çözüm |
|---|---|
| `CUDA out of memory` | Bu kart 8 GB; üç görev aynı anda sınırda çalışır (Görev 3 referans kaydında geçici OOM uyarısı görülebilir, sistem toparlıyor). `nvidia-smi` ile artık süreç kaldı mı bak, öldür, tekrar başlat. |
| `Could not import module 'AutoImageProcessor'` | Pillow eski. `pip3 install --user "pillow>=10.4"` |
| SLAM cmake'te `cannot find the CUDA libraries` | CUDA Toolkit 12.2 kurulu değil → `KURULUM.md` §0 |
| İlk karede ~30 sn duraklama | Normal (SLAM sözlüğü yükleniyor). |
| Derlemede apt kilidi hatası | Başka bir apt/dpkg işlemi sürüyor; bitmesini bekle, `build.sh`'i tekrar çalıştır. |

## 7) Yer kaplayan, silinebilir dosyalar

| Yol | Boyut | Not |
|---|---|---|
| `downloads/libtorch.zip` | 2,4 GB | Zaten `/usr/local/libtorch`'a açıldı; çevrimdışı paket istemiyorsan silinebilir. |
| `prova_yerel/` | ~190 MB | Yerel prova kareleri; `prova_verisi_uret.py` ile yeniden üretilir. |
| `~/Masaüstü/test/.../TAKIM_BAGLANTI_ARAYUZU/_images/` | değişken | İndirilen oturum kareleri. |
| `~/Pangolin`, `~/SP_SLAM3/build` | ~2 GB | Derleme ara çıktıları; SLAM binary'leri kurulduktan sonra gerekmez (yeniden derleyeceksen tut). |
