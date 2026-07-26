# VİDEO KONUŞMA METNİ — hamidiye_4907501

**TEKNOFEST 2026 Havacılıkta Yapay Zeka** · Teslim: 30 Temmuz 2026, 17:00 TSİ
**Yükleyen:** takım kaptanı · **Seslendiren:** takım üyesi · **Biçim:** ekran kaydı + dış ses

> ⚠ **Bu klasörün kökü** (çekim yaparken):
> `/home/kurt/hyz_YEDEK_2026-07-26/teslim`
> Aşağıdaki bütün yollar **bu köke göredir.** Yani `gorev2_engine.py` demek
> `/home/kurt/hyz_YEDEK_2026-07-26/teslim/gorev2_engine.py` demektir.
> Videoda **teslim ettiğimiz dosyaların ta kendisini** gösteriyoruz.

> **Bu belgeyi nasıl okuyacaksın:** `>` ile başlayan satırlar **söylenecek**
> metindir. `EKRAN #n` satırları ve 💡 notları **söylenmez**, sadece ne
> göstereceğini anlatır.

> ⚠ **Süre:** mail "5 dakikayı aşmayacak" diyor. Bu metin **620 konuşulan kelime**
> (sayıldı, tahmin değil). 150 kelime/dk → **4:08** · 145 → 4:17 ·
> **en yavaş ihtimalde (135) 4:36**. Her koşulda 5 dakikanın altında.

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
| 9 | 2:45 | **Görsel (tam ekran)** — BOZUK hizalama, 33.8 m | `gorseller/01_hizalama_BOZUK_33.8m.png` |
| 10 | 2:55 | **Görsel (tam ekran)** — DÜZELTİLMİŞ, 6.2 m | `gorseller/02_hizalama_DUZELTILMIS_6.2m.png` |
| 11 | 3:15 | Kod — FP koruması, **satır 275-295** | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |
| 12 | 3:30 | **Görsel (tam ekran)** — yeşil G3 kutusu + referans küçük resmi | `gorseller/06_uc_gorev_kare_1955.jpg` |
| 13 | 3:50 | Kod — `--drop`, `--limit` argümanları | `resmi_mock.py` |
| 14 | 4:00 | **Görsel (tam ekran)** — yörünge + hata eğrisi | `gorseller/04_2026_rgb_yorunge_ve_hata.png` |
| 15 | 4:15 | Kod — `detect()` tamamı, uzaklaştırılmış | `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` |

**Yedek görseller** (metinde geçmiyor, elinin altında dursun):
- RGB 3 eksen, GT vs SLAM: `gorseller/03_2026_rgb_3eksen_GT_vs_SLAM.png`
- Kutulu hareketli prova, 16 sn: `gorseller/07_kutulu_prova_16sn.mp4`

Klasörleri açmak için:
```bash
xdg-open ~/hyz_YEDEK_2026-07-26/teslim/gorseller
xdg-open ~/hyz_YEDEK_2026-07-26/teslim
```

---

# BÖLÜM A — ÇALIŞMA YAPISI

## 0:00 – 0:15 · Açılış

**EKRAN #1:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`
**satır 1-18** (modül başlığı, üç görevin listelendiği yer)

> Merhaba, hamidiye_4907501 takımıyız.
>
> Bu videoda size gönderdiğimiz kodu anlatacağız. Önce sistemin yapısı, sonra her
> görevde seçtiğimiz yöntem.

## 0:15 – 0:50 · İki süreç, bir köprü

**EKRAN #2:** aynı dosya **satır 32-43** (yol tanımları + üç import)
**EKRAN #3:** `gorev2_engine.py` **satır 149-159**
(`subprocess.Popen` ile SLAM'in başlatıldığı yer)

> Sistem resmî arayüzün main dosyasıyla başlıyor. O dosyaya hiç dokunmadık.
>
> Kurallar tek bir dosyayı değiştirmemize izin veriyordu. Biz de ekrandaki
> object detection model dosyasını değiştirdik. Burada ağır bir kod yok. Bu dosya
> sadece üç görevin motorunu çağırıyor. Orkestra şefi gibi düşünün.
>
> Program çalışırken iki süreç var. Biri Python süreci. Görev 1 ve Görev 3'ün
> modelleri burada, ikisi de aynı GPU'da. Diğeri C++ süreci. SLAM orada çalışıyor.
>
> Peki bu ikisi nasıl haberleşiyor? Dosya sistemi üzerinden. Python kareyi bir
> klasöre yazıyor. C++ tarafı bulduğu konumu bir metin dosyasına yazıyor. Python
> da o dosyayı sürekli takip ediyor.

## 0:50 – 1:25 · Bir karenin yolculuğu

**EKRAN #4:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`
→ `detect()` **satır 181-297**, yavaş kaydır. Üç görev bloğunun başlıkları ve
`try/except` satırları görünsün

> Şimdi tek bir karenin sistem içinde izlediği yolu takip edelim.
>
> Önce kareyi sunucudan indiriyoruz. Sonra ekrandaki detect fonksiyonu çalışıyor.
> Kareyi diskten sadece bir kez okuyoruz. Aynı görüntüyü hem Görev 1'e hem Görev
> 2'ye veriyoruz. Üç görev sırayla çalışıyor, hepsi sonucunu aynı tahmin nesnesine
> yazıyor. En sonunda tahmin sunucuya gidiyor.
>
> Bu döngüde kritik bir kısıt var. Sunucu, biz tahmin göndermeden bir sonraki
> kareyi vermiyor. Yani tek bir karede takılırsak bütün oturumu kaybediyoruz.
>
> O yüzden her görevi kendi try-except bloğuna aldık. Görev 1 çökerse kare boş
> tespitle gidiyor. Görev 2 çökerse ölü hesap konumuyla gidiyor. Ama kare mutlaka
> gidiyor. Bu bloklar hatayı gizlemek için değil. Döngü hiç durmasın diye.

---

# BÖLÜM B — ÇÖZÜM YAKLAŞIMLARI

## 1:25 – 2:00 · Görev 1, nesne tespiti

**EKRAN #5:** `gorev_1/yarisma_pipeline.py` → `kare_isle()` **satır 146-190**,
üç `_det()` çağrısını sırayla işaretle
**EKRAN #6 (tam ekran, ~1:45):** `gorseller/05_uc_gorev_kare_520.jpg`

> Görev 1 nesne tespiti. Modelimiz YOLO26-large, kendi verimizle eğittik. Dört
> sınıf var, yaklaşık 34 bin görüntü kullandık. mAP değerimiz 0.849.
>
> Ama tek geçişli bir tespit bu iş için yetmiyor. Üç kademeli bir hat kurduk.
>
> Birinci kademede 1280 pikselle bütün kareyi tarıyoruz.
>
> İkinci kademe iniş pedleri için. Aday pedi kırpıp modele bir daha soruyoruz.
> Buna hakem diyoruz.
>
> Üçüncü kademede onayladığımız pedin içine bakıyoruz. İnsan varsa o pede
> inilemez diyoruz.
>
> Hareketli mi sabit mi sorusunda da kameranın kendi hareketini ORB ve RANSAC ile
> çıkarıyoruz.

## 2:00 – 3:15 · Görev 2, GPS olmadan konum ⭐ VİDEONUN MERKEZİ

**EKRAN #7:** `gorev2_engine.py` → `process_frame()` **satır 297-340**
(sağlık 1 / sağlık 0 dallanması)
**EKRAN #8:** `alignment.py` → `_umeyama_2d()` **satır 112-121**
(yansıma adayı bloğu)
**EKRAN #9 (tam ekran, ~2:45):** `gorseller/01_hizalama_BOZUK_33.8m.png`
— başlıkta **"33.8 m"** görünsün, **3 saniye dur**
**EKRAN #10 (tam ekran, ~2:55):** `gorseller/02_hizalama_DUZELTILMIS_6.2m.png`
— başlıkta **"6.2 m"**

💡 **KURGU NOTU (okunmaz, sadece kurgu talimatı):** #9'dan #10'a kesme yapmadan,
üst üste bindirerek geç. İki grafik aynı oturum, aynı SLAM çıktısı; tek fark
hizalama matematiği. Yumuşak geçiş, farkı gözle vurur.

> Görev 2, GPS olmadan konum bulma.
>
> SLAM tarafında ORB-SLAM3'ün kendi geliştirdiğimiz sürümünü kullanıyoruz. Klasik
> ORB yerine SuperPoint ve LightGlue koyduk.
>
> Sunucu her karede bize bir sağlık biti gönderiyor. Sağlık bir ise gerçek konum
> zaten elimizde. Onu aynen geri gönderiyoruz, hatamız sıfır. Ama o kareyi SLAM'e
> de veriyoruz. Böylece SLAM'in bulduğu konumla gerçek konumu eşleştirip çiftler
> biriktiriyoruz.
>
> Sağlık sıfıra düştüğünde, yani kör bölgede, bu çiftlerden öğrendiğimiz dönüşümle
> SLAM'in konumunu metreye çeviriyoruz.
>
> En kritik bulgumuz burada. Standart çözüm Umeyama dönüşümü. Ama Umeyama
> yansımaya izin vermez.
>
> Bizim verimizde SLAM'in çerçevesiyle gerçek çerçeve arasında bir yansıma vardı.
> Yansımaya izin vermeyen çözüm x ve y'yi doğru buluyordu. Ama karşılığında z'yi
> ters çeviriyordu.
>
> Ekrandaki iki grafik aynı oturum, aynı SLAM çıktısı. Tek fark hizalama
> matematiği.
>
> Solda z ekseni tamamen ters gidiyor. Ortalama hata 33,8 metre.
>
> Sağda x ile y'yi yansımaya izin vererek hizaladık. Z'yi de ayrı çözdük. Hata 6,2
> metreye düştü. Z eksenindeki hata 27,7 metreden 3,2 metreye indi.
>
> Yarışmanın kendi metriğiyle 2025 oturumlarında hatamız 4,3 ile 8,8 metre
> arasında çıkıyor.

## 3:15 – 3:45 · Görev 3, referans nesne

**EKRAN #11:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`
**satır 275-295**, `if not bbox: continue` satırını fareyle işaretle
**EKRAN #12 (tam ekran, ~3:30):** `gorseller/06_uc_gorev_kare_1955.jpg`

> Görev 3 referans nesne tespiti. Burada bir sınıf aramıyoruz. Size verilen tek
> bir örnek fotoğraftaki o nesneyi bulmanız gerekiyor.
>
> Hibrit bir yöntem kurduk. FastSAM kareyi parçalara ayırıyor. DINOv2 her parçanın
> vektörünü çıkarıyor. Sonra referans fotoğrafla benzerliğine bakıyoruz.
>
> Sağdaki küçük resim aranan referans, yeşil kutu da sistemin bulduğu yer.
>
> Ekranda işaretlediğim kontrol bizim için kritik. Emin değilsek hiçbir kutu
> göndermiyoruz. Çünkü yanlış pozitif ceza getiriyor. Arayüzde güven skoru alanı
> yok. Gönderilen her kutu kesin iddia sayılıyor.

## 3:45 – 4:15 · Bu sayıları nasıl ölçtük

**EKRAN #13:** `resmi_mock.py` (`--drop`, `--limit` argümanları görünsün)
**EKRAN #14 (tam ekran, ~4:00):** `gorseller/04_2026_rgb_yorunge_ve_hata.png`
— **sağ paneldeki hata eğrisinde kare 1200'ü fareyle göster** (hatanın sıfıra düştüğü yer)

> Peki bu sayıları nereden biliyoruz?
>
> Resmî sunucunun yerel bir kopyasını yazdık. Aynı protokol, aynı sağlık biti
> düşüşleri, aynı referans pencereleri. Yarışma komutunun aynısını, sadece adresi
> değiştirerek yüzlerce kez çalıştırdık.
>
> Sağdaki grafik resmî 2026 örnek videosunun tamamı. Hata kör bölgede birikiyor.
> Sonra sistem altmış karelik sağlıklı bir pencere görünce hata sıfıra düşüyor.
> Sistem kendini yeniden hizalıyor.

## 4:15 – 4:30 · Kapanış

**EKRAN #15:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py`
→ `detect()` tamamı, uzaklaştırılmış

> Özetle şunu kurduk. Resmî arayüze tek dosyayla bağlanan, üç görevi tek süreçte
> çalıştıran, hiçbir hatada durmayan bir sistem.
>
> Canlı simülasyonda 2250 karenin hepsini gönderdik. Teşekkürler.

---

# ÇEKİM NOTLARI

## Ekran kaydı
- **1080p** kaydedin, 4K gerekmiyor.
- **Editörde yazı tipini 16-18 pt yapın.** Okunmayan kod, gösterilmemiş sayılır.
  En sık yapılan hata bu.
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
620 kelime. Prova çekiminde **4:45'i geçiyorsa** kesilecek ilk yerler, sırayla:
1. Görev 1'deki ego-hareket cümlesi ("Hareketli mi sabit mi…") — 14 kelime
2. "Bu sayıları nasıl ölçtük" ilk paragrafı — 28 kelime
3. Görev 1'deki "hakem" cümlesi ("Aday pedi kırpıp…") — 12 kelime

## Videoda söylenen her rakamın kaynağı
Bu tablo **kendi doğrulamanız için**: bir sayı söylemeden önce kaynağına bakın.

| Rakam | Kaynak |
|---|---|
| mAP@0.5 = 0.849 | Görev 1 eğitim kaydı |
| 33,8 m → 6,2 m (z: 27,7 → 3,2) | `gorseller/01_*.png` ve `02_*.png` grafik başlıkları |
| 2025 Denklem-2: 4,3 – 8,8 m | `evaluate_denklem2.py` ile hesaplanan oturum sonuçları |
| 2250/2250 kare | 16 Temmuz 2026 çevrim içi oturum kaydı |
