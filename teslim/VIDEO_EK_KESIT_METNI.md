# EK KESİT — "Üç görev örnek veri seti üzerinde çalışıyor"

Bu kesit, TEKNOFEST'in ek şartını karşılamak için mevcut videoya **eklenecek**
bölümdür:

> "Her görevin örnek veri seti üzerinde **çalıştırıldığı bir kesite** videoda
> yer verilmelidir."

**Klip:** `gorseller/08_uc_gorev_canli_kosu.mp4` (1920×1080, 25 fps, sessiz)
**Süre:** **29,8 saniye**
**Nereye:** Görev 3 bölümü bittikten sonra, "Bu sayıları nasıl ölçtük"
bölümünden **hemen önce** (mevcut videoda ~3:45 civarı)

## Klipte ne var (zaman çizelgesi)

| Klip zamanı | Ekranda |
|---|---|
| 0:00 – 0:08 | Başlık, veri seti/referans/kod yolları, profil satırı, model yüklemeleri |
| 0:08 – 0:20 | `SAGLIKLI` kareler — G2 hatası 0.0 m (gerçek konum görülüyor) |
| 0:20 – 0:23 | **`KESINTI`'ye geçiş** — kare 510, GPS kesiliyor |
| 0:23 – 0:26 | **Görev 3 devreye giriyor** — `G3:2 Ref01 Ref04`, sonra 4 referansa çıkıyor |
| 0:26 – 0:30 | ÖZET: 120 kare, G1 285 kutu, G2 kesintide ort. 0.6 m, G3 90 kutu |

---

## Söylenecek metin (78 kelime ≈ 31 saniye)

**▶ EKRAN:** `gorseller/08_uc_gorev_canli_kosu.mp4` — tam ekran oynat

> Şimdi üç görevi resmî örnek veri seti üzerinde birlikte çalıştırıyoruz.
>
> Ekrandaki komut, yarışma istemcisinin kullandığı detect fonksiyonunun aynısını
> çağırıyor. Tek fark, kareleri sunucudan değil örnek videodan okuması.
>
> Her satır bir kare. Görev 1'in bulduğu nesneler, Görev 2'nin konumu ve hatası,
> Görev 3'ün referans kutuları aynı satırda.
>
> Üstteki kareler sağlıklı, sistem gerçek konumu görüyor. Aşağıda kesintiye
> giriyoruz, GPS kesiliyor. Görev 2 artık kendi kestirimini üretiyor.
>
> Referans penceresi açıldığında Görev 3 de kutu göndermeye başlıyor.
>
> Üç görev, tek süreçte, her karede tam bir tahmin.

---

## Kurgu notları

- Klibi **hızlandırmadan** oynat; satırların akışı okunabilir olmalı.
- Kesitin başında model yükleme satırları var (`[1/3] … hazir`). Bunlar
  "sistem gerçekten ayağa kalkıyor" izlenimi verdiği için **kesme**.
- `SAGLIKLI` → `KESINTI` geçişini gösteren an kritik: o satırda bir saniye
  bekle, anlatımda da oraya denk getir.
- Klibin sonunda ÖZET satırı var (toplam kutu, ortalama hata). Anlatımın son
  cümlesi ona denk gelirse iyi olur.

## Süre bütçesi

Mevcut video **4:58.5**, sınır 5:00. Bu kesit ~30 saniye eklediği için
**mevcut videodan ~30 saniye kesmen gerekiyor.**

Kesilmesi önerilen yer: mevcut videoda **2:00–2:30 arası**, `gorev2_engine.py`
içindeki uzun kod kaydırma bölümü. Orada anlatım devam ederken görüntü fazla
değişmiyor; kaydırmayı kısaltmak anlamı bozmaz.

> ⚠ Kesin kesme noktasını sen belirlemelisin — hangi cümlenin nereye denk
> geldiğini ses üzerinden kontrol etmen gerekiyor.

## Bu kesit hangi şartları karşılıyor

| Mailin şartı | Nasıl karşılanıyor |
|---|---|
| Her görevin örnek veri üzerinde çalıştırıldığı kesit | Üç görev de aynı ekranda, aynı koşuda, kare kare |
| Girdi formatı | Örnek videodan okunan kare (1920×1080, 7.5 kare/sn) |
| Çıktı formatı | Satırda görünüyor: G1 kutu sayısı+sınıf, G2 xyz+hata, G3 kutu+referans adı |
| Bileşenlerin entegrasyonu | Tek süreç, tek `detect()` çağrısı — satırın üçü birden dolması bunu gösteriyor |
| Video ile kodun uyumu | Çalışan betik `demo_ornek_veri.py`, teslim paketinin içinde |
