# Konuşma metni burada değil

Geçerli ve **tek** konuşma metni şurada:

    ../teslim/VIDEO_KONUSMA_METNI.md

Tam yol:

    /home/kurt/hyz_YEDEK_2026-07-26/teslim/VIDEO_KONUSMA_METNI.md

## Neden taşındı

Metindeki bütün dosya yolları artık **teslim klasörüne göre göreli**
(`gorev2_engine.py`, `gorseller/05_uc_gorev_kare_520.jpg` gibi).
Böylece videoda jüriye gönderdiğimiz dosyaların ta kendisini gösteriyoruz,
çalışma klasöründeki kopyaları değil.

Bu klasörde duran 4 eski taslak (çalışma klasörü yollarıyla yazılmıştı)
27 Temmuz 2026'da silindi — karışıklığa yol açıyorlardı.
Gerekirse git geçmişinden geri alınabilir:

    git log --diff-filter=D --name-only -- video_metinleri/
