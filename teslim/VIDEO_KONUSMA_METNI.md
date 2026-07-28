# VİDEO KONUŞMA METNİ — hamidiye_4907501

**TEKNOFEST 2026 Havacılıkta Yapay Zeka** · Teslim: 30 Temmuz 2026, 17:00 TSİ
**Yükleyen:** takım kaptanı · **Seslendiren:** takım üyesi · **Biçim:** ekran kaydı + dış ses

> ⚠ **Bu klasörün kökü** (çekim yaparken):
> `/home/kurt/hyz_YEDEK_2026-07-26/teslim`
> Aşağıdaki bütün yollar **bu köke göredir.** Yani `gorev2_engine.py` demek
> `/home/kurt/hyz_YEDEK_2026-07-26/teslim/gorev2_engine.py` demektir.
> Videoda **teslim ettiğimiz dosyaların ta kendisini** gösteriyoruz.

> **Bu belgeyi nasıl okuyacaksın:** `>` ile başlayan satırlar **söylenecek**
> metindir. `▶ EKRAN` / `▶ FARE` satırları ve 💡 notları **söylenmez**, sadece ne
> göstereceğini anlatır.

> ⚠ **Süre:** mail "5 dakikayı aşmayacak" diyor. Bu metin **706 konuşulan kelime.**
> **Ölçülen tempon: 131 kelime/dk** (27 Temmuz videosu: 650 kelime → 4:58.5).
> Ham süre bu tempoda **≈ 5:28** → **hızlandırma şart.**

> 🎬 **KURGUDA NE DEĞİŞTİ (27 Temmuz videosuna göre)**
> **Eklenen:** (a) #14 canlı koşu klibi, 18 sn — TEKNOFEST'in "her görevin örnek
> veri seti üzerinde çalıştırıldığı kesit" şartı için; (b) kapanışta KURULUM.md /
> README.md'ye yönlendiren cümle. **Hiçbir şey çıkarılmadı.**
>
> **Süre çözümü — hızlandırma.** Ham video ≈ 5:28. Gereken çarpan:
> | Çarpan | Sonuç |
> |---|---|
> | 1.08x | 5:04 ❌ hâlâ aşıyor |
> | **1.11x** | **4:55 ✅ önerilen** |
> | 1.15x | 4:45 (biraz aceleci duyulur) |
>
> Sesi tizleştirmeden hızlandırmak için (`atempo` perdeyi korur):
> ```bash
> ffmpeg -i ham_video.mp4 -filter_complex \
>   "[0:v]setpts=PTS/1.11[v];[0:a]atempo=1.11[a]" \
>   -map "[v]" -map "[a]" -c:v libx264 -crf 20 -preset slow \
>   -c:a aac -b:a 192k -movflags +faststart nihai_video.mp4
> ```
> **Kurgudan sonra süreyi mutlaka ölç:** `ffprobe -show_entries format=duration nihai_video.mp4`
>
> Hızlandırma istemezsen alternatif: klibi 18 sn yerine 12 sn'ye kırp ve
> "Bu sayıları nasıl ölçtük" paragrafını çıkar → ~5:02, yine de sınırda kalır.

---

# 📋 ÇEKİM LİSTESİ — sırayla açılacak dosyalar

Çekimden önce bu dosyaları sekmelerde/klasörde hazır aç. Sıra aynen bu.

| # | Zaman | Ne gösterilecek | YOL (teslim klasörüne göre) |
|---|---|---|---|
| 1 | 0:00 | Kod — modül başlığı, **satır 1-18** | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |
| 2 | 0:15 | Aynı dosya, **satır 32-43** (yollar + import) | *(yukarıdakinin aynısı)* |
| 3 | 0:30 | Kod — SLAM'in başlatılması, **satır 149-159** | `gorev2_engine.py` |
| 4 | 0:50 | Kod — `detect()`, **satır 181-297**, yavaş kaydır | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |
| 5 | 1:25 | Kod — `kare_isle()`, **satır 146-190** | `gorev_1/yarisma_pipeline.py` |
| 6 | 1:45 | **Görsel (tam ekran)** — turuncu Görev 1 kutuları | `gorseller/05_uc_gorev_kare_520.jpg` |
| 7 | 2:00 | Kod — `process_frame()`, **satır 297-340** | `gorev2_engine.py` |
| 8 | 2:15 | Kod — `_umeyama_2d()`, **satır 112-121** | `alignment.py` |
| 9 | 2:40 | **Görsel (tam ekran)** — eksen bazında hata (x:4.0 y:2.2 **z:32.0**) | `gorseller/03_2026_rgb_3eksen_GT_vs_SLAM.png` |
| 10 | 2:45 | **Görsel (tam ekran)** — BOZUK hizalama, 33.8 m | `gorseller/01_hizalama_BOZUK_33.8m.png` |
| 11 | 2:55 | **Görsel (tam ekran)** — DÜZELTİLMİŞ, 6.2 m | `gorseller/02_hizalama_DUZELTILMIS_6.2m.png` |
| 12 | 3:15 | Kod — FP koruması, **satır 275-295** | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |
| 13 | 3:30 | **Görsel (tam ekran)** — yeşil G3 kutusu + referans küçük resmi | `gorseller/06_uc_gorev_kare_1955.jpg` |
| 14 | 3:45 | **KLİP (tam ekran, 18 sn)** — üç görev örnek veri üzerinde çalışıyor | `gorseller/08_uc_gorev_canli_kosu.mp4` |
| 15 | 4:05 | Kod — `--drop`, `--limit` argümanları | `resmi_mock.py` |
| 16 | 4:15 | **Görsel (tam ekran)** — yörünge + hata eğrisi | `gorseller/04_2026_rgb_yorunge_ve_hata.png` |
| 17 | 4:35 | Dosya ağacı — `KURULUM.md` ve `README.md`'yi göster | *(teslim klasörü kökü)* |
| 18 | 4:45 | Kod — `detect()` tamamı, uzaklaştırılmış | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |

**Yedek görseller** (metinde geçmiyor, elinin altında dursun):
- Kutulu hareketli prova, 16 sn: `gorseller/07_kutulu_prova_16sn.mp4`

Klasörleri açmak için:
```bash
xdg-open ~/hyz_YEDEK_2026-07-26/teslim/gorseller
xdg-open ~/hyz_YEDEK_2026-07-26/teslim
```

---

# BÖLÜM A — ÇALIŞMA YAPISI

## 0:00 – 0:15 · Açılış

**▶ EKRAN:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`,
**satır 1-18** (modül başlığı, üç görevin listelendiği yer)

> Merhaba, hamidiye_4907501 takımıyız.
>
> Bu videoda size gönderdiğimiz kodu anlatacağız. Önce sistemin yapısı, sonra her
> görevde seçtiğimiz yöntem.

## 0:15 – 0:50 · İki süreç, bir köprü

**▶ EKRAN:** aynı dosyada **satır 32-43**'e in (yol tanımları + üç import)

> Sistem resmî arayüzün main dosyasıyla başlıyor. O dosyaya hiç dokunmadık.
>
> Kurallar tek bir dosyayı değiştirmemize izin veriyordu. Biz de ekrandaki
> object detection model dosyasını değiştirdik. Burada ağır bir kod yok. Bu dosya
> sadece üç görevin motorunu çağırıyor. Orkestra şefi gibi düşünün.

**▶ EKRAN DEĞİŞ:** `gorev2_engine.py`, **satır 149-159**
*`Ctrl+G` → 149 yaz, oraya git. **Satır 155'teki `subprocess.Popen(` ifadesini
fareyle seç** — C++ sürecinin doğduğu yer orası. "Diğeri C++ süreci" derken imleç
o satırda olsun.*

> Program çalışırken iki süreç var. Biri Python süreci. Görev 1 ve Görev 3'ün
> modelleri burada, ikisi de aynı GPU'da. Diğeri C++ süreci. SLAM orada çalışıyor.

**▶ FARE:** **satır 152**'yi seç — `cmd = [binp, self.vocab, self.settings, self.inbox, self.outbox]`.
*`inbox` ve `outbox` kelimelerinin üstünde dur; iki sürecin buluştuğu klasörler bunlar.*

> Peki bu ikisi nasıl haberleşiyor? Dosya sistemi üzerinden. Python kareyi bir
> klasöre yazıyor. C++ tarafı bulduğu konumu bir metin dosyasına yazıyor. Python
> da o dosyayı sürekli takip ediyor.

## 0:50 – 1:25 · Bir karenin yolculuğu

**▶ EKRAN DEĞİŞ:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`,
`detect()` başı (**satır 181**)

> Şimdi tek bir karenin sistem içinde izlediği yolu takip edelim.
>
> Önce kareyi sunucudan indiriyoruz. Sonra ekrandaki detect fonksiyonu çalışıyor.
> Kareyi diskten sadece bir kez okuyoruz. Aynı görüntüyü hem Görev 1'e hem Görev
> 2'ye veriyoruz. Üç görev sırayla çalışıyor, hepsi sonucunu aynı tahmin nesnesine
> yazıyor. En sonunda tahmin sunucuya gidiyor.
>
> Bu döngüde kritik bir kısıt var. Sunucu, biz tahmin göndermeden bir sonraki
> kareyi vermiyor. Yani tek bir karede takılırsak bütün oturumu kaybediyoruz.

**▶ FARE:** yavaşça aşağı kaydır ve **üç `try:` satırını sırayla göster**:
*satır 204 civarı (Görev 1) → satır 225 civarı (Görev 2) → satır 282 civarı (Görev 3).
Her birinin altındaki `except Exception as e:` satırını da göster; "çökerse"
kelimesini söylerken imleç `except` satırında olsun.*

> O yüzden her görevi kendi try-except bloğuna aldık. Görev 1 çökerse kare boş
> tespitle gidiyor. Görev 2 çökerse ölü hesap konumuyla gidiyor. Ama kare mutlaka
> gidiyor. Bu bloklar hatayı gizlemek için değil. Döngü hiç durmasın diye.

---

# BÖLÜM B — ÇÖZÜM YAKLAŞIMLARI

## 1:25 – 2:00 · Görev 1, nesne tespiti

**▶ EKRAN:** `gorev_1/yarisma_pipeline.py`, `kare_isle()` görünsün (satır 146)

> Görev 1 nesne tespiti. Modelimiz YOLO26-large, kendi verimizle eğittik. Dört
> sınıf var, yaklaşık 34 bin görüntü kullandık. mAP değerimiz 0.849.
>
> Ama tek geçişli bir tespit bu iş için yetmiyor. Üç kademeli bir hat kurduk.

**▶ FARE:** **satır 149**'u seç — `for s, c, x1, y1, x2, y2 in _det(kare, 1280, ...)`.
*`1280` sayısının üstünde dur, cümlede o sayıyı söylüyorsun.*

> Birinci kademede 1280 pikselle bütün kareyi tarıyoruz.

**▶ FARE:** **satır 164**'e in — `zc = max([cc for ss, cc, *_ in _det(crop, 192, 0.10)...`
*Önce satır 159'daki `_kirp(...)` çağrısını göster (ped kırpılıyor), sonra 164'teki
`192` sayısını göster (kırpılan parça modele yeniden soruluyor).*

> İkinci kademe iniş pedleri için. Aday pedi kırpıp modele bir daha soruyoruz.
> Buna hakem diyoruz.

**▶ FARE:** **satır 168**'e in — `for si, ci, a1, b1, a2, b2 in _det(crop, 640, 0.10)`.
*`640`'ı göster, sonra satır 183'e in: `inis = 0 if (dolu or kenar or ...) else 1`.
"insan varsa inilemez" derken imleç `dolu` değişkeninde olsun.*

> Üçüncü kademede onayladığımız pedin içine bakıyoruz. İnsan varsa o pede
> inilemez diyoruz.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/05_uc_gorev_kare_520.jpg`
*Sağdaki kırmızı aracın turuncu kutusunu fareyle göster, sonra üstündeki
**"tasit sabit"** etiketini göster — "sabit" kelimesini söylerken imleç orada olsun.
Sol alttaki beyaz minibüsün etiketi de aynı şeyi diyor, istersen ona da geç.*

> Hareketli mi sabit mi sorusunda da kameranın kendi hareketini ORB ve RANSAC ile
> çıkarıyoruz.

## 2:00 – 3:15 · Görev 2, GPS olmadan konum ⭐ VİDEONUN MERKEZİ

**▶ EKRAN:** `gorev2_engine.py`, `process_frame()` (satır 297 civarı)

> Görev 2, GPS olmadan konum bulma.
>
> SLAM tarafında ORB-SLAM3'ün kendi geliştirdiğimiz sürümünü kullanıyoruz. Klasik
> ORB yerine SuperPoint ve LightGlue koyduk.

**▶ FARE:** dallanmayı sırayla göster: **satır 308** (`if int(health) == 1 ...`) →
**satır 311** (`kaynak = "echo"`) → **satır 325-336** (kör bölge, `tf.apply(...)`) →
**satır 336** (`kaynak = "slam"`) → **satır 339** (`kaynak = "deadreckon"`).
*Üç kaynak adını (`echo` / `slam` / `deadreckon`) tek tek göstermen yeterli;
anlattığın üç durum tam bunlar.*

> Sunucu her karede bize bir sağlık biti gönderiyor. Sağlık bir ise gerçek konum
> zaten elimizde. Onu aynen geri gönderiyoruz, hatamız sıfır. Ama o kareyi SLAM'e
> de veriyoruz. Böylece SLAM'in bulduğu konumla gerçek konumu eşleştirip çiftler
> biriktiriyoruz.
>
> Sağlık sıfıra düştüğünde, yani kör bölgede, bu çiftlerden öğrendiğimiz dönüşümle
> SLAM'in konumunu metreye çeviriyoruz.

**▶ EKRAN DEĞİŞ:** `alignment.py`, **satır 112-121** (`allow_reflection` bloğu)
*`Ctrl+G` → 112. **Satır 117'deki `S_refl[1, 1] = -1.0` satırını fareyle seç ve
seçili bırak** — anlattığın yansıma tam orası, "yansıma vardı" derken imleç orada
olsun. Sonra satır 119-120'yi (`if cand[3] < best[3]`) göster: artığı küçük olan
adayın seçildiği yer.*

> En kritik bulgumuz burada. Standart çözüm Umeyama dönüşümü. Ama Umeyama
> yansımaya izin vermez.
>
> Bizim verimizde SLAM'in çerçevesiyle gerçek çerçeve arasında bir yansıma vardı.
> Yansımaya izin vermeyen çözüm x ve y'yi doğru buluyordu. Ama karşılığında z'yi
> ters çeviriyordu.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/03_2026_rgb_3eksen_GT_vs_SLAM.png`
*Görseli tam ekran aç. **Önce en üstteki başlık satırında `eksen MAE x:4.0 y:2.2
z:32.0` yazan yeri fareyle göster.** Sonra sırayla: üst panelde (x) iki çizginin
çakışık olduğunu göster, orta panelde (y) aynısını, en alt panelde (z) yeşil
çizginin maviden aşağı doğru ayrıldığı yeri göster. Üç panel arası geçişte
acele etme.*

> Bunu eksen eksen ayırdığımızda görüyoruz. Ekranda x ve y çizgileri çakışık,
> ortalama hataları dört ve iki metre. Z ise otuz iki metre sapıyor. Yani hata
> neredeyse tamamen z ekseninde.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/01_hizalama_BOZUK_33.8m.png`
*Cümleyi söylemeden önce **1 saniye sus**, göz grafiğe otursun.*

> Şimdi göreceğiniz iki grafik aynı oturum, aynı SLAM çıktısı. Tek fark hizalama
> matematiği.
>
> Bu ilk grafikte z ekseni tamamen ters gidiyor. Ortalama hata 33,8 metre.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/02_hizalama_DUZELTILMIS_6.2m.png`

💡 **KURGU NOTU (okunmaz):** Bu geçişi kesme yapmadan, üst üste bindirerek yap.
İki grafiğin ekseni aynı; yumuşak geçiş, yeşil çizginin maviye oturuşunu gözle
vurur. Geçiş bitince cümleye başla.

> Şimdi düzeltilmiş hali. X ile y'yi yansımaya izin vererek hizaladık. Z'yi de
> ayrı çözdük. Hata 6,2 metreye düştü. Z eksenindeki hata 27,7 metreden 3,2
> metreye indi.
>
> Yarışmanın kendi metriğiyle 2025 oturumlarında hatamız 4,3 ile 8,8 metre
> arasında çıkıyor.

## 3:15 – 3:45 · Görev 3, referans nesne

**▶ EKRAN:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`,
**satır 275** civarı (Görev 3 bloğunun başı)

> Görev 3 referans nesne tespiti. Burada bir sınıf aramıyoruz. Size verilen tek
> bir örnek fotoğraftaki o nesneyi bulmanız gerekiyor.
>
> Hibrit bir yöntem kurduk. FastSAM kareyi parçalara ayırıyor. DINOv2 her parçanın
> vektörünü çıkarıyor. Sonra referans fotoğrafla benzerliğine bakıyoruz.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/06_uc_gorev_kare_1955.jpg`
*Sırayı bozma, akış şu: **(1)** sağ üstteki `Ref 02` küçük resmini fareyle göster
("aranan referans" bu) → **(2)** karenin ortasındaki büyük yeşil kutuya götür
(halı saha — bulunan yer) → **(3)** altındaki `Ref 03` küçük resmini göster →
**(4)** aşağıdaki `Ref 03` yeşil kutusuna (kale direği) götür.
Yani her referansı **önce aranan fotoğraf, sonra bulunan yer** diye eşleştir.*

> Sağdaki küçük resim aranan referans, yeşil kutu da sistemin bulduğu yer.

**▶ EKRAN DEĞİŞ:** aynı kod dosyasına dön, **satır 292**'yi fareyle seç
(`if not bbox: continue`)

> Ekranda işaretlediğim kontrol bizim için kritik. Emin değilsek hiçbir kutu
> göndermiyoruz. Çünkü yanlış pozitif ceza getiriyor. Arayüzde güven skoru alanı
> yok. Gönderilen her kutu kesin iddia sayılıyor.

## 3:45 – 4:05 · Üç görev örnek veri seti üzerinde ⬅ YENİ

**▶ EKRAN DEĞİŞ:** `gorseller/08_uc_gorev_canli_kosu.mp4` — **tam ekran oynat (18 sn)**

💡 **KURGU NOTU (okunmaz):** Klip sessizdir, üzerine konuşulur, hızlandırma.
İç zamanlaması: 0-4,5 sn başlık + model yüklemeleri · 4,5-14,5 sn `SAGLIKLI`
kareler → `KESINTI` geçişi → Görev 3 kutuları beliriyor · 14,5-18 sn özet.
Anlatım 31 kelime ≈ 14 sn, yani **sonda ~4 saniye sessizlik kalır** — özet
satırı okunsun diye bilerek böyle.

> Şimdi üç görevi resmî örnek veri seti üzerinde çalıştırıyoruz. Her satır bir
> kare: Görev 1'in nesneleri, Görev 2'nin konumu ve hatası, Görev 3'ün referans
> kutuları. Kesintide Görev 2 kendi kestirimini üretiyor.

## 4:05 – 4:35 · Bu sayıları nasıl ölçtük

**▶ EKRAN DEĞİŞ:** `resmi_mock.py`, **satır 177-180** (`--limit`, `--drop` argümanları)

> Peki bu sayıları nereden biliyoruz?
>
> Resmî sunucunun yerel bir kopyasını yazdık. Aynı protokol, aynı sağlık biti
> düşüşleri, aynı referans pencereleri. Yarışma komutunun aynısını, sadece adresi
> değiştirerek yüzlerce kez çalıştırdık.

**▶ EKRAN DEĞİŞ (tam ekran):** `gorseller/04_2026_rgb_yorunge_ve_hata.png`
*İki panel var: **solda** kuşbakışı uçuş yolu (mavi gerçek, yeşil SLAM),
**sağda** hata eğrisi. Görsel açılınca 1 saniye sol panelde dur (yeşilin maviyi
takip ettiği görünsün), sonra **imleci sağ panele götür** ve orada kal.*

> Ekrandaki grafik resmî 2026 örnek videosunun tamamı.

**▶ FARE:** sağ panelde imleci **kare 450'den 1150'ye doğru yavaşça sürükle** —
eğrinin sıfırdan tırmanıp ~48 metreye çıktığı turuncu bölge burası.

> Sağ panelde hata kör bölgede birikiyor.

**▶ FARE:** sağ panelde **kare 1200**'deki dik düşüşü göster — eğri bir anda
sıfıra iniyor. *İmleci tam o dikey inişin üstüne koy ve orada 2 saniye bekle.*

> Sonra sistem altmış karelik sağlıklı bir pencere görünce hata sıfıra düşüyor.
> Sistem kendini yeniden hizalıyor.

## 4:35 – 4:55 · Kapanış

**▶ EKRAN DEĞİŞ:** dosya ağacında **`KURULUM.md`** ve **`README.md`**'yi göster
(tek tıkla açıp içindekiler kısmını 2 saniye göstermen yeterli)

> Kurulumun tamamı KURULUM ve README dosyalarında adım adım yazılı. Kurmada ya da
> çalıştırmada sorun yaşarsanız bu iki dosyaya bakabilirsiniz.

**▶ EKRAN DEĞİŞ:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`,
`detect()` tamamı. `Ctrl+-` ile 2-3 kez uzaklaştır, fonksiyon bütün olarak görünsün.

> Özetle şunu kurduk. Resmî arayüze tek dosyayla bağlanan, üç görevi tek süreçte
> çalıştıran, hiçbir hatada durmayan bir sistem.
>
> Canlı simülasyonda 2250 karenin hepsini gönderdik. Teşekkürler.

---

# ÇEKİM NOTLARI

## Ekran kaydı
- **1080p** kaydedin, 4K gerekmiyor.
- **Editörde yazı tipini 16-18 pt yapın.** Okunmayan kod, gösterilmemiş sayılır.
- Dosyaları **önceden sekmelerde açın** (çekim listesi sırasıyla).
- Anlattığınız satırı fareyle seçin ya da kısa zoom yapın.
- Grafikleri **tam ekran** gösterin, kod sekmesinin yanında küçük değil.
- **Sesi ayrı kaydedip üstüne bindirin** — ekran kaydı sırasında konuşursanız
  klavye ve fan sesi girer.

## Doğal konuşmak için
- **Metni ekrandan okumayın.** Paragrafı okuyun, kapatın, kendi cümlelerinizle
  söyleyin. Metin kelime kelime değil, fikir sırası olarak doğru.
- Rakamları söylerken hafif yavaşlayın (0.849, 33,8 metre, 6,2 metre).
- Grafiğe geçtiğinizde **bir saniye susun.** Göz grafiğe otursun, sonra konuşun.
- "Şurada", "burası" gibi belirsiz sözler kullanmayın. Ya adını söyleyin
  ("detect fonksiyonu"), ya konumunu ("soldaki grafik", "ekranda işaretlediğim satır").

## Kadraja ASLA girmemesi gerekenler
- Kimlik bilgisi dosyası (`config/.env`) — takım adı, **şifre**, sunucu adresi.
  *Not: bu dosya teslim paketinde YOK, sadece `example.env` şablonu var.*
- Terminal geçmişinde kimlik bilgisi geçen komutlar
- Tarayıcı sekmeleri, masaüstü bildirimleri (kayıttan önce kapatın)

## Süre kontrolü
706 kelime, ham süre ≈ 5:28. **1.11x hızlandırma ile 4:55.** Hızlandırmak istemezsen
kesilecek ilk yerler, sırayla:
1. Görsel 03 cümlesini kısalt → "Hata neredeyse tamamen z ekseninde; x ve y
   çakışık." (görseli göster ama uzun anlatma) — ~18 kelime kazanır
2. Görev 1'deki "hakem" cümlesi ("Aday pedi kırpıp…") — 12 kelime
3. Kapanıştaki "Canlı simülasyonda…" cümlesi — 8 kelime

## Videoda söylenen her rakamın kaynağı
Bu tablo **kendi doğrulamanız için**: bir sayı söylemeden önce kaynağına bakın.

| Rakam | Kaynak |
|---|---|
| mAP@0.5 = 0.849 | Görev 1 eğitim kaydı |
| x:4.0 y:2.2 z:32.0 m | `gorseller/03_*.png` grafik başlığı |
| 33,8 m → 6,2 m (z: 27,7 → 3,2) | `gorseller/01_*.png` ve `02_*.png` grafik başlıkları |
| 2025 Denklem-2: 4,3 – 8,8 m | `evaluate_denklem2.py` ile hesaplanan oturum sonuçları |
| 2250/2250 kare | 16 Temmuz 2026 çevrim içi oturum kaydı |
| Klipteki özet (120 kare, 285 / 0.6 m / 90) | `demo_ornek_veri.py` koşusu, 27 Temmuz |
