# 5 DAKİKALIK KOD ANLATIM VİDEOSU — KONUŞMA METNİ (TASLAK)

**Takım:** hamidiye_4907501 · TEKNOFEST 2026 Havacılıkta Yapay Zeka

> **Not:** Mail görüntüsü elime ulaşmadığı için istenen format (dil, yüz görünmesi,
> kimlerin konuşacağı, özel içerik şartları) hakkında varsayım yaptım:
> tek anlatıcı, Türkçe, ekran kaydı üzerine ses. Mailde farklı şart varsa metni
> ona göre kırpalım.
>
> **Uzunluk:** 687 konuşulan kelime. Ölçüm: 130 kelime/dk → 5:17 · 145 kelime/dk
> → 4:44 · 160 kelime/dk → 4:18. Yani normal anlatım temposunda 5 dakikaya
> oturuyor; hedef 5:00 için **~140 kelime/dk** tempoyu tut. Aşağıdaki zaman
> damgaları bu tempoya göre.

---

## 0:00 – 0:20 · GİRİŞ

**EKRAN:** README.md'nin en üstü (başlık + 2250/2250 satırı) ya da yarisma_yorunge.png

> Merhaba, TEKNOFEST 2026 Havacılıkta Yapay Zeka yarışmasında hamidiye_4907501
> takımıyız. Bu videoda üç görevi tek istemcide birleştiren sistemimizin kodunu
> anlatacağız. 16 Temmuz simülasyonunda bu kod 2250 karenin tamamını 53 buçuk
> dakikada işleyip gönderdi — sıfır ret, sıfır kopma.

---

## 0:20 – 0:55 · MİMARİ VE TEMEL İLKE

**EKRAN:** README.md'deki mimari şeması → sonra `object_detection_model.py`
`detect()` fonksiyonu (üç görev bloğunun yorum satırları görünsün)

> Mimari şöyle: yarışmanın resmî bağlantı arayüzünü koruduk, kurallara uygun
> olarak yalnızca `object_detection_model.py` dosyasını değiştirdik. Bu dosya
> orkestra şefi: kareyi diskten bir kez okuyup aynı diziyi üç göreve de veriyor —
> 4K kareyi üç kez decode etmek pahalı olurdu.
>
> Kodun her yerine yayılmış tek bir ilke var. Sunucu, biz tahmin göndermeden
> sonraki kareyi vermiyor; yani tek karede takılmak bütün oturumu kaybettirir.
> Bu yüzden hangi görev hata verirse versin her kareye tam bir geçerli tahmin
> gidiyor. Kodda gördüğünüz try-except blokları hata yutmak için değil, kareyi
> ilerletmek için.

---

## 0:55 – 1:40 · GÖREV 1 — NESNE TESPİTİ

**EKRAN:** `gorev_1/yarisma_pipeline.py` → `kare_isle()` fonksiyonu; üç `_det()`
çağrısını (1280 / 192 / 640) sırayla göster

> Görev 1, nesne tespiti. Çekirdekte kendi verimizle eğittiğimiz YOLO26-large
> var: dört sınıf, yaklaşık 34 bin görüntü, mAP@0.5 değeri 0.849.
>
> Ama tek geçiş yetmiyor. `yarisma_pipeline.py`'de üç kademe kurduk. Bir: 1280
> pikselle tam kare taraması. İki: iniş pedi adaylarını çok düşük eşikle
> topluyoruz, sonra her adayın etrafını kırpıp modele tekrar soruyoruz — "hakem"
> dediğimiz kademe. 4K karede küçücük görünen bir ped, kırpılıp büyütülünce çok
> daha güvenilir sınıflanıyor. Üç: onaylanan pedin içinde insan taraması
> yapıyoruz; insan varsa iniş durumu "inilemez" oluyor.
>
> Hareketli-sabit ayrımında asıl sorun kameranın kendisinin hareket etmesi. Bunu
> ORB öznitelikleri ve RANSAC homografisiyle çözdük: kareler arası kamera
> hareketini kestirip iz merkezlerini ona göre taşıyoruz, kalan hareket gerçek
> nesne hareketi oluyor. Hattın hızı 4K karede p95 70 milisaniye.

---

## 1:40 – 3:05 · GÖREV 2 — GPS'SİZ KONUM KESTİRİMİ ⭐ (en uzun bölüm)

**EKRAN:** `gorev2_engine.py` `process_frame()` → sonra `alignment.py`
`solve_alignment()` → sonra `rapor_o2/1_kusbakisi_yorunge.png` (yörünge grafiği)

> Görev 2, GPS'siz konum kestirimi — projenin en zorlu kısmı buydu.
>
> Temelimiz ORB-SLAM3'ün kendi çattığımız bir sürümü: klasik ORB yerine
> SuperPoint ve LightGlue kullanıyor. C++ tarafı ayrı bir süreç olarak çalışıyor;
> Python köprüsü kasıtlı olarak çok basit — bir inbox klasörüne kareyi atomik
> yazıyoruz, C++ tarafı `pose.txt`'e pozu yazıyor, biz de o dosyayı bir iş
> parçacığıyla tail ediyoruz. Kareleri PNG olarak yazıyoruz, JPEG değil: kalite
> 95 JPEG bile SuperPoint'i zayıflatıp izleme kaybı tetikliyordu.
>
> Sunucu bize her karede bir sağlık biti veriyor. Sağlık 1'de gerçek konum zaten
> elimizde; onu aynen geri gönderiyoruz, hatası sıfır. Ama aynı kareyi SLAM'e de
> besleyip "SLAM konumu – gerçek konum" çiftleri biriktiriyoruz. Sağlık 0'a
> düştüğünde, yani kör bölgede, bu çiftlerden öğrendiğimiz dönüşümle SLAM'in
> ölçeksiz konumunu metreye çeviriyoruz.
>
> İşte projenin en önemli keşfi burada. Klasik Umeyama çözümü determinantı artı
> bire zorlar, yani yansımaya izin vermez. Bizim ölçümlerimizde SLAM çerçevesiyle
> gerçek çerçeve arasında bir yansıma vardı; yansımasız çözüm x-y'yi doğru
> eşleyip z'yi ters çeviriyordu. `alignment.py`'de bunu yeniden yazdık: x-y
> düzleminde yansımaya izin verilen iki boyutlu hizalama, artı z için ayrı tek
> boyutlu ölçek-öteleme. Tek bu değişiklik kör bölge hatasını 48 metreden 7
> metreye indirdi.
>
> İki koruma daha var: SLAM yeni bir harita açarsa eski hizalamayı çöpe atıyoruz,
> çünkü artık geçersiz. Ve poz hiç gelmezse sabit hızla ilerletiyoruz — ama üstel
> sönümle, uzun kesintide sınırsız savrulmayı engellemek için.
>
> Sonuç: 2025 oturumlarında yarışmanın Denklem-2 metriğiyle 4,3 ile 8,8 metre
> arası hata; ilk sürümümüze göre yaklaşık sekiz kat iyileşme.

---

## 3:05 – 3:45 · GÖREV 3 — REFERANS NESNE TESPİTİ

**EKRAN:** `tam_prova_gorseller/kare_1954_uc_gorev.jpg` (kutular görünür kare) →
`object_detection_model.py` içindeki Görev 3 bloğu (`if not bbox: continue` satırı)

> Görev 3, referans nesne tespiti: burada sınıf değil, size verilen tek bir örnek
> görüntüdeki o belirli nesneyi bulmanız isteniyor. Hibrit çözdük — FastSAM kareyi
> bölütlere ayırıyor, DINOv2 her bölütün gömme vektörünü çıkarıyor, referans
> görüntünün gömmesiyle kosinüs benzerliği hesaplıyoruz. Termal görüntülerde
> semantik gömme zayıfladığı için oraya ELoFTR eşleştirme ve MAGSAC doğrulaması
> ekledik.
>
> Buradaki kritik tasarım kararı şu: yanlış pozitif cezalı ve güven skoru alanı
> yok — gönderdiğiniz her kutu kesin iddia sayılıyor. Bu yüzden sistem emin
> olmadığı karede hiç kutu göndermiyor. Simülasyonda yedi referansın altısında
> isabetli kutular ürettik.

---

## 3:45 – 4:30 · DOĞRULAMA — NEYE DAYANARAK GÜVENİYORUZ

**EKRAN:** `resmi_mock.py` → `analiz_3eksen/oturum2.png` → `SETUP_LOG.md`'de hızlı kaydırma

> Peki bu sisteme neye dayanarak güveniyoruz? Yarışma günü tek şansınız var, o
> yüzden resmî sunucunun yerel bir taklidini yazdık: `resmi_mock.py`. Aynı HTTP
> protokolü, aynı sağlık biti düşüşleri, aynı referans pencereleri. Böylece
> yarışma komutunun ta kendisini, sadece adresi değiştirerek yüzlerce kez
> koşturabildik. Yanına canlı izleme panelleri, üç eksenli hata grafikleri ve
> otomatik rapor üreticileri ekledik.
>
> Bütün karar günlüğünü `SETUP_LOG.md`'de tuttuk — işe yaramayan denemeler dahil.
> Örneğin SLAM'e ölü hesap harmanlama fikri hatayı 11 metreden 44 metreye
> çıkardı; kodda kapalı duruyor ama gerekçesiyle birlikte duruyor.

---

## 4:30 – 5:00 · KAPANIŞ

**EKRAN:** README depo haritası tablosu ya da üç depo adı

> Özetle: resmî arayüze tek dosyayla bağlanan, üç görevi aynı GPU'da tek süreçte
> çalıştıran ve hiçbir hatada durmayan bir sistem. Canlı simülasyonda 2250
> karenin tamamı gönderildi. Kaynak kod ve kurulum belgeleri üç depoda; temiz bir
> makinede tek komutla kurulabiliyor. İzlediğiniz için teşekkürler.

---

# ÇEKİM NOTLARI

## Tempo
- Metin dakikada ~145 kelimeye ayarlı. Prova çekiminde 5:30'u geçiyorsan
  **kesilecek ilk yerler:** Görev 3'ün termal cümlesi, doğrulama bölümündeki
  SETUP_LOG paragrafı.
- 4:30'un altında kalıyorsan **eklenecek yer:** aşağıdaki "yedek içerik".

## Yedek içerik (süre artarsa)
- **SLAM init nokta tavanı:** LightGlue ~1600 noktanın üstünde bozuluyordu;
  init çıkarıcısını 1200'e sınırladık, "init'te takılma" tamamen bitti
  (`Tracking.cc`).
- **Kamera otomatiği:** kalibrasyon dosyası ilk karenin genişliğinden otomatik
  seçiliyor (4000→4K, 3840→cropA, 1920→1080p, 640→termal).
- **Eski koşu zehirlenmesi:** `run_yarisma` klasörü siliniyor değil, zaman
  damgasıyla kenara taşınıyor — denetim izi kalsın, eski kare numaraları yeni
  koşuyla eşleşip hizalamayı bozmasın.
- **2026 örnek videodaki fiziksel sınır:** ~35 m hatanın %98'i z ekseninde;
  kalibrasyon dakikası düz uçuşsa z ölçeği gözlemlenemiyor. xy hatası 5,4 m.

## Söylerken dikkat
- "Çelik zırh" ilkesini bir kez net söyle; jüri için en ayırt edici tasarım
  kararı bu.
- Görev 2'deki yansıma keşfi videonun en güçlü teknik anı — orada hızlanma.
- Rakamları yuvarlamadan söyle (0.849, 70 ms, 4,3–8,8 m); belgelerde birebir
  doğrulanabiliyor.
- Kimlik bilgisi/şifre içeren hiçbir ekran görüntüsü almayın: `config/.env`
  dosyasını ve `_logs` altındaki token satırlarını kadraja sokmayın.
