# KONUŞMA METNİ #2 — JÜRİ TESLİMİ (mail şartlarına göre)

**Takım:** hamidiye_4907501 · TEKNOFEST 2026 Havacılıkta Yapay Zeka
**Teslim:** 30 Temmuz 2026 Perşembe 17:00 TSİ · T3 KYS → Başvurularım → Yeni Form Atandı
**Yükleyen:** takım danışmanı **veya** takım kaptanı (takım üyeleri yükleyemiyor)

## Mailin istediği şey, birebir

> "Kaynak kodlarının **çalışma yapısını** ve **çözüm yaklaşımlarını** açıklayan,
> **5 dakikayı aşmayacak** video kaydı."
>
> "Gönderilen kaynak kodlar ve video kayıtları üzerinde gerçekleştirilecek
> **teknik incelemelerin** ardından finalist takımların kesin listesi ilan edilecektir."

Bu metin buna göre kurgulandı — 1. metinden farkı:

| | Metin #1 | **Metin #2 (bu)** |
|---|---|---|
| Amaç | tanıtım / sonuç vurgusu | mailin iki şartı: **çalışma yapısı** + **çözüm yaklaşımları** |
| Yapı | görev görev | **A) yapı → B) yöntemler** (mailin sırası) |
| Ekran | README dahil | **README YOK** — sadece kod + grafik |
| Dil | yazı dili | **konuşma dili** — kısa cümle, okuyormuş gibi durmuyor |

## Bu metin nasıl yazıldı (okumadan önce oku)

Cümleler bilerek kısa. Noktalı virgül yok, ara cümle yok, "—" ile eklenen
açıklama yok. Çünkü bunlar yazıda güzel durur ama ağızdan çıkarken **metinden
okuduğun anlaşılır.**

- Her cümlede tek fikir var. Nefes noktaları doğal geliyor.
- Bazı cümleler devrik. Konuşurken zaten öyle konuşuyoruz.
- "Şimdi", "peki", "yani", "işte" gibi bağlaçlar bilerek bırakıldı. Bunlar
  metni doğallaştırıyor, silme.
- **Ezberleme.** Bir paragrafı oku, kapat, kendi cümlelerinle söyle. Metin
  kelime kelime değil, sıra ve fikir olarak doğru.

> ⚠ **Süre uyarısı:** mail "aşmayacak" diyor. 5:00'i geçen video reddedilebilir.
> Bu metin **666 konuşulan kelime**, ortalama cümle uzunluğu 6.2 kelime. Kısa
> cümleler hızlı akar, bu üslupta tempo genelde 150 kelime/dk olur: **4:26.**
> 145'te 4:35. Ağır konuşursan (135) 4:56'ya çıkar — o zaman aşağıdaki
> "süre kontrolü" kesintilerini uygula. Kurguda hedef **4:30 civarı.**

---

# BÖLÜM A — ÇALIŞMA YAPISI

## 0:00 – 0:15 · Açılış

**EKRAN:** `object_detection_model.py` en üstü — modül başlığındaki üç görev
listesi görünsün (satır 1-18)

> Merhaba, hamidiye_4907501 takımıyız.
>
> Bu videoda size gönderdiğimiz kodu anlatacağız. Önce sistem nasıl çalışıyor,
> ona bakalım. Sonra her görevde hangi yöntemi neden seçtiğimize geçeceğiz.

## 0:15 – 0:50 · İki süreç, bir köprü

**EKRAN:** `object_detection_model.py` satır 32-43 (yol tanımları + üç import) →
`gorev2_engine.py` satır 149-159 (`subprocess.Popen` ile SLAM'in başlatılması)

> Sistem resmî arayüzün main dosyasıyla başlıyor. O dosyaya hiç dokunmadık.
>
> Kurallar tek bir dosyayı değiştirmemize izin veriyordu. Biz de şu an ekranda
> gördüğünüz dosyayı değiştirdik. Burada ağır bir kod yok. Bu dosya sadece üç
> görevin motorunu çağırıyor. Orkestra şefi gibi düşünün.
>
> Program çalışırken iki süreç var. Biri Python süreci. Görev 1 ve Görev 3'ün
> modelleri burada, ikisi de aynı GPU'da. Diğeri C++ süreci. SLAM orada çalışıyor.
>
> Peki bu ikisi nasıl haberleşiyor? Dosya sistemi üzerinden. Bunu bilerek böyle
> seçtik. Python kareyi bir klasöre yazıyor. C++ tarafı bulduğu konumu bir metin
> dosyasına yazıyor. Python da o dosyayı sürekli takip ediyor.

## 0:50 – 1:30 · Bir karenin yolculuğu

**EKRAN:** `detect()` fonksiyonunu yavaş kaydır (satır 181-297). Üç görev
bloğunun başlıkları ve `try/except`ler görünsün

> Bir kare şuradan geçiyor.
>
> Önce kareyi sunucudan indiriyoruz. Sonra detect fonksiyonu çalışıyor. Burada
> kareyi diskten sadece bir kez okuyoruz. Aynı görüntüyü hem Görev 1'e hem
> Görev 2'ye veriyoruz. Çünkü 4K bir kareyi iki kez açmak bize pahalıya patlıyor.
> Üç görev sırayla çalışıyor. Hepsi sonucunu aynı tahmin nesnesine yazıyor.
>
> Şimdi burada çok kritik bir kısıt var. Sunucu, biz tahmin göndermeden bir
> sonraki kareyi vermiyor. Yani tek bir karede takılırsak bütün oturumu
> kaybediyoruz.
>
> O yüzden her görevi kendi try-except bloğuna aldık. Görev 1 çökerse kare boş
> tespitle gidiyor. Görev 2 çökerse ölü hesap konumuyla gidiyor. Ama kare mutlaka
> gidiyor. Yani bu bloklar hatayı gizlemek için değil. Döngü hiç durmasın diye.

---

# BÖLÜM B — ÇÖZÜM YAKLAŞIMLARI

## 1:30 – 2:10 · Görev 1, nesne tespiti

**EKRAN:** `gorev_1/yarisma_pipeline.py` → `kare_isle()` (satır 146-190).
Üç `_det()` çağrısını sırayla işaretle → `tam_prova_gorseller/kare_1954_uc_gorev.jpg`

> Görev 1 nesne tespiti. Modelimiz YOLO26-large, kendi verimizle eğittik. Dört
> sınıf var, yaklaşık 34 bin görüntü kullandık. mAP değerimiz 0.849.
>
> Ama tek geçişli bir tespit bu iş için yetmiyor. Biz üç kademeli bir hat kurduk.
> Ekranda üç ayrı çağrıyı görüyorsunuz.
>
> Birinci kademede 1280 pikselle bütün kareyi tarıyoruz.
>
> İkinci kademe iniş pedleri için. Aday pedi kırpıp modele bir daha soruyoruz.
> Buna hakem diyoruz. Çünkü 4K karede minicik görünen bir ped, kırpıp büyütünce
> çok daha net anlaşılıyor.
>
> Üçüncü kademede onayladığımız pedin içine bakıyoruz. İnsan varsa o pede
> inilemez diyoruz.
>
> Bir de şu var: hareketli mi, sabit mi? Buradaki asıl problem kameranın kendisinin
> hareket etmesi. Biz ORB ve RANSAC ile kameranın hareketini hesaplayıp çıkarıyoruz.
> Geriye kalan gerçek hareket oluyor. Bu hattın hızı 4K karede 70 milisaniye.

## 2:10 – 3:30 · Görev 2, GPS olmadan konum ⭐ VİDEONUN MERKEZİ

**EKRAN sırası (bu bölümde kurgu önemli):**
1. `gorev2_engine.py` `process_frame()` — sağlık 1 / sağlık 0 dallanması (satır 297-340)
2. `alignment.py` `_umeyama_2d()` — yansıma adayı bloğu (satır 112-121)
3. `analiz_3eksen/oturum4.png` **tam ekran** (z ekseni ters, 33.8 m)
4. `analiz_3eksen/oturum4_duzeltilmis.png` (6.2 m) — 3'ten 4'e **geçiş yaparak**

> Görev 2, GPS olmadan konum bulma.
>
> SLAM tarafında ORB-SLAM3'ün kendi geliştirdiğimiz sürümünü kullanıyoruz. Klasik
> ORB yerine SuperPoint ve LightGlue koyduk.
>
> Mantık şöyle işliyor. Sunucu her karede bize bir sağlık biti gönderiyor. Sağlık
> bir ise gerçek konum zaten elimizde. Onu aynen geri gönderiyoruz, hatamız sıfır.
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
> **(grafiğe geç)** Şu iki grafiğe bakalım. İkisi de aynı oturum. Aynı SLAM
> çıktısı. Tek fark hizalama matematiği.
>
> Solda z ekseni tamamen ters gidiyor. Ortalama hata 33,8 metre.
>
> Sağda x ve y'yi yansımaya izin vererek hizaladık. Z'yi de ayrı çözdük. Hata 6,2
> metreye düştü. Z eksenindeki hata 27,7 metreden 3,2 metreye indi.
>
> Yarışmanın kendi metriğiyle ölçtüğümüzde hatamız üç oturumda 4,3 ile 8,8 metre
> arasında çıkıyor.

## 3:30 – 4:00 · Görev 3, referans nesne

**EKRAN:** `object_detection_model.py` satır 275-295. `if not bbox: continue`
satırını işaretle → `gorev3_test_cikti/rgb_frame08019.jpg`

> Görev 3 referans nesne tespiti. Burada bir sınıf aramıyoruz. Size verilen tek
> bir örnek fotoğraftaki o nesneyi bulmanız gerekiyor.
>
> Biz hibrit bir yöntem kurduk. FastSAM kareyi parçalara ayırıyor. DINOv2 her
> parçanın vektörünü çıkarıyor. Sonra referans fotoğrafla benzerliğine bakıyoruz.
>
> Koddaki şu satır bizim için kritik. Emin değilsek hiçbir kutu göndermiyoruz.
> Çünkü yanlış pozitif ceza getiriyor. Arayüzde de güven skoru alanı yok.
> Gönderdiğiniz her kutu kesin iddia sayılıyor.

## 4:00 – 4:25 · Bu sayıları nasıl ölçtük

**EKRAN:** `resmi_mock.py` (`--drop`, `--limit` argümanları) →
`gorev2_engine.py` satır 354-357 (CSV yazımı, `kaynak` kolonu) → `evaluate_denklem2.py`

> Peki bu sayıları nereden biliyoruz?
>
> Resmî sunucunun yerel bir kopyasını yazdık. Aynı protokol, aynı sağlık biti
> düşüşleri, aynı referans pencereleri. Böylece yarışma komutunun aynısını, sadece
> adresi değiştirerek yüzlerce kez çalıştırdık.
>
> Motor her karenin konumunu ve o konumu nereden aldığını CSV'ye yazıyor. Puanı
> da doğrudan bu dosyadan hesaplıyoruz.

## 4:25 – 4:40 · Kapanış

**EKRAN:** `detect()` fonksiyonunun tamamı, uzaklaştırılmış görünümde

> Özetle şunu kurduk. Resmî arayüze tek dosyayla bağlanan, üç görevi tek süreçte
> çalıştıran, hiçbir hatada durmayan bir sistem.
>
> Canlı simülasyonda 2250 karenin hepsini gönderdik. Teşekkürler.

---

# ÇEKİM NOTLARI

## Doğal konuşmak için
- **Metni ekrandan okuma.** Paragrafı oku, kapat, kendi cümlelerinle söyle. Bu
  metin kelime kelime değil, fikir sırası olarak doğru.
- Rakamları söylerken hafif yavaşla. 0.849, 70 milisaniye, 33,8 metre gibi
  yerlerde jürinin yazması gerekebilir.
- Grafiğe geçtiğinde bir saniye sus. Göz grafiğe otursun, sonra konuş.
- İlk çekimde tutmazsa bölüm bölüm çek. Zaten kurguda birleştireceksin.

## Ekran kaydı hazırlığı
- Editörde yazı tipini büyüt (14-16 pt). Okunmayan kod, gösterilmemiş sayılır.
- Anlattığın satırı fareyle işaretle ya da kısa zoom yap.
- Dosyaları önceden sekmelerde aç. Video sırasında dosya arama.

## Kadraja ASLA girmemesi gerekenler
- `istemci/TAKIM_BAGLANTI_ARAYUZU/config/.env` (takım adı, şifre, sunucu adresi)
- `kanitlar/yarisma_oturum_logu_temiz.log` içindeki token satırları
- Terminal geçmişinde kimlik bilgisi geçen komutlar

## Süre kontrolü
- 666 kelime. Prova çekiminde **4:45'i geçiyorsa** kesilecek ilk yerler:
  1. "Bu sayıları nasıl ölçtük" bölümünün son paragrafı — 17 kelime
  2. Görev 1'deki "hakem" gerekçesi ("Çünkü 4K karede minicik…") — 15 kelime
  3. Görev 1'deki ego-hareket paragrafı ("Bir de şu var…") — 37 kelime
- 4:10'un altında kalıyorsan aşağıdaki yedek içerikten ekle.

## Yedek içerik (süre artarsa)
- **Görev 3 termal yolu** (süre için çıkarıldı, ilk bunu ekle): "Termal
  görüntülerde bu vektörler zayıflıyor. Oraya ELoFTR ve MAGSAC ekledik." — 15 kelime
- **PNG kararı:** "Kareleri SLAM'e PNG olarak veriyoruz, JPEG değil. Kaliteli bir
  JPEG bile SuperPoint'i zayıflatıp izlemeyi kaybettiriyordu."
- **Kamera otomatiği:** "Kalibrasyon dosyasını ilk karenin genişliğinden otomatik
  seçiyoruz. 4000 ise 4K, 640 ise termal."
- **Harita sıfırlaması:** "SLAM yeni bir harita açarsa eski hizalamayı çöpe
  atıyoruz. Çünkü artık geçerli değil."

## ⚠ Teslim öncesi düzeltilmesi gereken tutarsızlık
`rapor_o2/RAPOR.md` **v1 (eski baz)** sonuçlarını içeriyor: O2 için 40.23 m.
Nihai konfigürasyon aynı oturumda 8.8 m — `SETUP_LOG.md` satır 572-578'deki v8
tablosunda. Jüri kaynak kodu inceleyeceği için bu dosyayı ya güncelleyin ya da
başına "v1 baz ölçümü, nihai sonuçlar için SETUP_LOG v8 tablosu" notu ekleyin.
`rapor_o2/*.png` grafikleri de v1 verisinden üretilmiş — **videoda onları
göstermeyin.** `analiz_3eksen/` altındakileri kullanın.

## Rakamların kaynağı (jüri sorarsa)
| Rakam | Nerede |
|---|---|
| mAP@0.5 = 0.849, p95 = 70 ms | `SETUP_LOG.md` satır 599-601 |
| 33.8 m → 6.2 m, z 27.7 → 3.2 m | `analiz_3eksen/oturum4*.png` grafik başlıkları |
| Denklem-2: O2 8.8 / O3 4.3 / O4 4.7 | `SETUP_LOG.md` satır 572-578 (v8 tablosu) |
| 2250/2250 kare | `kanitlar/yarisma_sent.jsonl` + oturum logu |
