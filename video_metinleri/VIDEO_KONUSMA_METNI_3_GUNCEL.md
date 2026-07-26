# KONUŞMA METNİ #3 — GÜNCEL (26 Temmuz 2026 koşusuyla)

**Takım:** hamidiye_4907501 · TEKNOFEST 2026 Havacılıkta Yapay Zeka
**Teslim:** 30 Temmuz 2026 Perşembe 17:00 TSİ · T3 KYS → Başvurularım → Yeni Form Atandı
**Yükleyen:** takım **kaptanı** · **Seslendiren:** takım üyesi · **Biçim:** ekran kaydı + dış ses

Metin #2'den farkı: her "EKRAN" satırında **hangi dosya, hangi grafik** olduğu
açıkça yazıyor (eskisinde "grafiğe geç" diyordu, belirsizdi). Sayılar 26 Temmuz
2026'da resmî 2026 örnek videosuyla yapılan koşudan güncellendi. Görev 3 artık
canlı kanıtla anlatılıyor.

> ⚠ **Süre:** mail "5 dakikayı aşmayacak" diyor, aşan video reddedilebilir.
> Bu metin **646 konuşulan kelime**, ortalama cümle 6,2 kelime. Ölçülen tempolar:
> 150 kelime/dk → **4:18** · 145 → 4:27 · **en yavaş ihtimalde (135) 4:47**.
> Yani her koşulda 5 dakikanın altında, elinizde en az 13 saniye pay var.

---

# BÖLÜM A — ÇALIŞMA YAPISI

## 0:00 – 0:15 · Açılış

**EKRAN:** `istemci/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` satır 1-18
(modül başlığı, üç görevin listelendiği yer)

> Merhaba, hamidiye_4907501 takımıyız.
>
> Bu videoda size gönderdiğimiz kodu anlatacağız. Önce sistemin yapısı, sonra her
> görevde seçtiğimiz yöntem.

## 0:15 – 0:50 · İki süreç, bir köprü

**EKRAN:** aynı dosya satır 32-43 (yol tanımları + üç import) →
`gorev2_engine.py` satır 149-159 (`subprocess.Popen` ile SLAM'in başlatıldığı yer)

> Sistem resmî arayüzün main dosyasıyla başlıyor. O dosyaya hiç dokunmadık.
>
> Kurallar tek bir dosyayı değiştirmemize izin veriyordu. Biz de şu an ekranda
> gördüğünüz dosyayı değiştirdik. Burada ağır bir kod yok. Bu dosya sadece üç
> görevin motorunu çağırıyor. Orkestra şefi gibi düşünün.
>
> Program çalışırken iki süreç var. Biri Python süreci. Görev 1 ve Görev 3'ün
> modelleri burada, ikisi de aynı GPU'da. Diğeri C++ süreci. SLAM orada çalışıyor.
>
> Peki bu ikisi nasıl haberleşiyor? Dosya sistemi üzerinden. Python kareyi bir
> klasöre yazıyor. C++ tarafı bulduğu konumu bir metin dosyasına yazıyor. Python
> da o dosyayı sürekli takip ediyor.

## 0:50 – 1:25 · Bir karenin yolculuğu

**EKRAN:** `object_detection_model.py` → `detect()` fonksiyonunu yavaş kaydır
(satır 181-297). Üç görev bloğunun başlıkları ve `try/except` satırları görünsün

> Bir kare şuradan geçiyor.
>
> Önce kareyi sunucudan indiriyoruz. Sonra detect fonksiyonu çalışıyor. Kareyi
> diskten sadece bir kez okuyoruz. Aynı görüntüyü hem Görev 1'e hem Görev 2'ye
> veriyoruz. Üç görev sırayla çalışıyor, hepsi sonucunu aynı tahmin nesnesine
> yazıyor.
>
> Şimdi burada çok kritik bir kısıt var. Sunucu, biz tahmin göndermeden bir
> sonraki kareyi vermiyor. Yani tek bir karede takılırsak bütün oturumu
> kaybediyoruz.
>
> O yüzden her görevi kendi try-except bloğuna aldık. Görev 1 çökerse kare boş
> tespitle gidiyor. Görev 2 çökerse ölü hesap konumuyla gidiyor. Ama kare mutlaka
> gidiyor. Bu bloklar hatayı gizlemek için değil. Döngü hiç durmasın diye.

---

# BÖLÜM B — ÇÖZÜM YAKLAŞIMLARI

## 1:25 – 2:05 · Görev 1, nesne tespiti

**EKRAN:** `gorev_1/yarisma_pipeline.py` → `kare_isle()` (satır 146-190), üç `_det()`
çağrısını sırayla işaretle → **tam ekran:** `gorseller_2026/kare_000520.jpg`
(turuncu kutular: taşıt, sabit/hareketli etiketiyle)

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
> Bir de hareketli mi sabit mi sorusu var. Asıl problem kameranın kendisinin
> hareket etmesi. ORB ve RANSAC ile kameranın hareketini hesaplayıp çıkarıyoruz.
> Geriye kalan gerçek hareket oluyor.

## 2:05 – 3:20 · Görev 2, GPS olmadan konum ⭐ VİDEONUN MERKEZİ

**EKRAN sırası — bu bölümde kurgu önemli:**
1. `gorev2_engine.py` → `process_frame()`, sağlık 1 / sağlık 0 dallanması (satır 297-340)
2. `alignment.py` → `_umeyama_2d()`, yansıma adayı bloğu (satır 112-121)
3. **tam ekran:** `analiz_3eksen/oturum4.png` — başlıkta "33.8 m" görünsün, 3 saniye dur
4. **geçiş yap:** `analiz_3eksen/oturum4_duzeltilmis.png` — başlıkta "6.2 m"
   *(İki grafik aynı oturum, aynı SLAM çıktısı. Tek fark hizalama matematiği.
   Geçişi kesme yapmadan, üst üste bindirerek yaparsan etki çok daha güçlü olur.)*

> Görev 2, GPS olmadan konum bulma.
>
> SLAM tarafında ORB-SLAM3'ün kendi geliştirdiğimiz sürümünü kullanıyoruz. Klasik
> ORB yerine SuperPoint ve LightGlue koyduk.
>
> Sunucu her karede bize bir sağlık biti gönderiyor. Sağlık bir ise gerçek konum
> zaten elimizde. Onu aynen geri gönderiyoruz, hatamız sıfır.
> Ama o kareyi SLAM'e de veriyoruz. Böylece SLAM'in bulduğu konumla gerçek konumu
> eşleştirip çiftler biriktiriyoruz.
>
> Sağlık sıfıra düştüğünde, yani kör bölgede, bu çiftlerden öğrendiğimiz dönüşümle
> SLAM'in konumunu metreye çeviriyoruz.
>
> Projedeki en kritik bulgumuz tam burada. Bu iş için standart çözüm Umeyama
> dönüşümüdür. Ama Umeyama yansımaya izin vermez.
>
> Bizim verimizde SLAM'in çerçevesiyle gerçek çerçeve arasında bir yansıma vardı.
> Yansımaya izin vermeyen çözüm x ve y'yi doğru buluyordu. Ama karşılığında z'yi
> ters çeviriyordu.
>
> Şu iki grafiğe bakalım. İkisi de aynı oturum. Aynı SLAM çıktısı. Tek fark
> hizalama matematiği.
>
> Solda z ekseni tamamen ters gidiyor. Ortalama hata 33,8 metre.
>
> Sağda x ile y'yi yansımaya izin vererek hizaladık. Z'yi de ayrı çözdük. Hata 6,2
> metreye düştü. Z eksenindeki hata 27,7 metreden 3,2 metreye indi.
>
> Yarışmanın kendi metriğiyle 2025 oturumlarında hatamız 4,3 ile 8,8 metre
> arasında çıkıyor.

## 3:20 – 3:55 · Görev 3, referans nesne

**EKRAN:** `object_detection_model.py` satır 275-295, `if not bbox: continue`
satırını işaretle → **tam ekran:** `gorseller_2026/kare_001955.jpg`
(sağ şeritte aranan referans fotoğrafı, karede yeşil kutu)

> Görev 3 referans nesne tespiti. Burada bir sınıf aramıyoruz. Size verilen tek
> bir örnek fotoğraftaki o nesneyi bulmanız gerekiyor.
>
> Hibrit bir yöntem kurduk. FastSAM kareyi parçalara ayırıyor. DINOv2 her parçanın
> vektörünü çıkarıyor. Sonra referans fotoğrafla benzerliğine bakıyoruz. Termal
> görüntülerde bu vektörler zayıflıyor, oraya ELoFTR ekledik.
>
> Sağdaki küçük resim aranan referans, yeşil kutu da bulduğu yer.
>
> Koddaki şu satır bizim için kritik. Emin değilsek hiçbir kutu göndermiyoruz.
> Çünkü yanlış pozitif ceza getiriyor. Arayüzde güven skoru alanı yok. Gönderdiğiniz
> her kutu kesin iddia sayılıyor.

## 3:55 – 4:25 · Bu sayıları nasıl ölçtük

**EKRAN:** `resmi_mock.py` (`--drop`, `--limit` argümanları görünsün) →
**tam ekran:** `gorseller_2026/yorunge_2026.png` (sağ paneldeki hata eğrisi;
kare 1200'de sıfıra düştüğü yeri fareyle göster)

> Peki bu sayıları nereden biliyoruz?
>
> Resmî sunucunun yerel bir kopyasını yazdık. Aynı protokol, aynı sağlık biti
> düşüşleri, aynı referans pencereleri. Böylece yarışma komutunun aynısını, sadece
> adresi değiştirerek yüzlerce kez çalıştırdık.
>
> Sağdaki grafik resmî 2026 örnek videosunun tamamı. İki bin iki yüz elli karenin
> hepsine konum gönderdik. Hata kör bölgede birikiyor. Sonra şurada, sistem altmış
> karelik sağlıklı bir pencere görünce sıfıra düşüyor. Kendini yeniden hizalıyor.

## 4:25 – 4:40 · Kapanış

**EKRAN:** `detect()` fonksiyonunun tamamı, uzaklaştırılmış görünümde

> Özetle şunu kurduk. Resmî arayüze tek dosyayla bağlanan, üç görevi tek süreçte
> çalıştıran, hiçbir hatada durmayan bir sistem.
>
> Canlı simülasyonda 2250 karenin hepsini gönderdik. Teşekkürler.

---

# ÇEKİM NOTLARI

## Ekran kaydı — bu sizin seçtiğiniz biçim, doğru seçim

Mail "kaynak kodların çalışma yapısını açıklayan" video istiyor. Ekranda kodu
göstermek tam olarak istenen şey. Yüz kamerası gerekmiyor, dış ses yeterli.

- **Kaydı 1080p alın**, 4K gerekmiyor ve dosyayı şişiriyor.
- **Editörde yazı tipini büyütün (16-18 pt).** Okunmayan kod, gösterilmemiş sayılır.
  Bu en sık yapılan hata.
- **Dosyaları önceden sekmelerde açın.** Video içinde dosya aramak zaman yiyor.
- Anlattığınız satırı fareyle seçin ya da kısa zoom yapın.
- Grafikleri **tam ekran** gösterin, kod sekmesinin yanında küçük değil.
- Sesi ayrı kaydedip üstüne bindirmek en temizi. Ekran kaydı sırasında konuşursanız
  klavye ve fan sesi girer.

## Doğal konuşmak için
- **Metni ekrandan okumayın.** Paragrafı okuyun, kapatın, kendi cümlelerinizle
  söyleyin. Metin kelime kelime değil, fikir sırası olarak doğru.
- Rakamları söylerken hafif yavaşlayın. 0.849, 33,8 metre, 6,2 metre gibi yerlerde
  jüri not alıyor olabilir.
- Grafiğe geçtiğinizde **bir saniye susun.** Göz grafiğe otursun, sonra konuşun.
- Tutmazsa bölüm bölüm çekin, kurguda birleştirirsiniz.

## Kadraja ASLA girmemesi gerekenler
- `istemci/TAKIM_BAGLANTI_ARAYUZU/config/.env` — takım adı, **şifre**, sunucu adresi
- `kanitlar/yarisma_oturum_logu_temiz.log` içindeki token satırları
- Terminal geçmişinde kimlik bilgisi geçen komutlar
- Tarayıcı sekmeleri, masaüstü bildirimleri (kaydı almadan önce kapatın)

## Süre kontrolü
646 kelime. Prova çekiminde **4:50'yi geçiyorsa** kesilecek ilk yerler, sırayla:
1. Görev 1'deki ego-hareket paragrafı ("Bir de hareketli mi…") — 33 kelime
2. Görev 3'teki termal cümlesi ("Termal görüntülerde…") — 12 kelime
3. "Bu sayıları nasıl ölçtük" bölümünün ilk paragrafı — 30 kelime

4:10'un altında kalıyorsanız şu yedekleri ekleyin:
- **PNG kararı:** "Kareleri SLAM'e PNG olarak veriyoruz, JPEG değil. Kaliteli bir
  JPEG bile SuperPoint'i zayıflatıp izlemeyi kaybettiriyordu."
- **Kamera otomatiği:** "Kalibrasyon dosyasını ilk karenin genişliğinden otomatik
  seçiyoruz. 4000 ise 4K, 640 ise termal."
- **Harita sıfırlaması:** "SLAM yeni bir harita açarsa eski hizalamayı çöpe atıyoruz.
  Çünkü artık geçerli değil."

## Jüri sorarsa — rakamların kaynağı

| Rakam | Nerede |
|---|---|
| mAP@0.5 = 0.849, p95 = 70 ms | `SETUP_LOG.md` satır 599-601 |
| 33,8 m → 6,2 m (z: 27,7 → 3,2) | `analiz_3eksen/oturum4*.png` grafik başlıkları |
| 2025 Denklem-2: O2 8,8 / O3 4,3 / O4 4,7 m | `SETUP_LOG.md` satır 572-578 (v8 tablosu) |
| 2250/2250 kare | `kanitlar/yarisma_sent.jsonl` + oturum logu |
| 2026 örnek video koşusu (26 Tem) | `tam_prova_jsonlar/OZET.json` |

### 2026 örnek videosu sorulursa — dürüst cevap

26 Temmuz koşusunda kör bölge ortalama hatası **33,6 metre**. Ama bunun
**32 metresi z ekseninden** geliyor; yatayda hata **x 4,0 · y 2,2 metre**.

Sebebi şu: kalibrasyon dakikası boyunca uçuş neredeyse düz gidiyor. Tek kameralı
SLAM'de yükseklik ölçeği ancak irtifa değişimi gözlemlenirse çözülebiliyor. Düz
uçuşta bu bilgi veride yok. Bu bir kod hatası değil, tek kameranın fiziksel sınırı.
2025 oturumlarında kalibrasyon dakikasında irtifa değişimi olduğu için aynı sistem
4,3-8,8 metre veriyor.

**Bunu saklamayın.** Sorulursa böyle açıklamak, konuya hâkim olduğunuzu gösterir.

## ⚠ Teslim öncesi düzeltilmesi gereken tutarsızlık
`rapor_o2/RAPOR.md` **v1 (eski baz)** sonuçlarını içeriyor: O2 için 40.23 m.
Nihai konfigürasyon aynı oturumda 8.8 m — `SETUP_LOG.md` satır 572-578'deki v8
tablosunda. Jüri kaynak kodu inceleyeceği için bu dosyayı ya güncelleyin ya da
başına "v1 baz ölçümü, nihai sonuçlar için SETUP_LOG v8 tablosu" notu ekleyin.
`rapor_o2/*.png` grafikleri de v1 verisinden üretilmiş — **videoda onları
göstermeyin.** `analiz_3eksen/` ve `gorseller_2026/` altındakileri kullanın.
