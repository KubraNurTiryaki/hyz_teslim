"""Esik kalibrasyonu - mAP@0.5'i maksimize eden esikleri bul (mAP'i ASIL belirleyen is).

Neden gerekli: payload'da guven skoru yok -> her gonderilen kutu kesin pozitif.
`samdino_min_cos` cok dusukse FP artar (AP duser), cok yuksekse recall duser. Dogru
esik yalnizca ETIKETLI mini-val ile bulunur.

Akis:
  1) GT JSON yukle (tools/label_minival.py ile uretilir).
  2) Her etiketli kare icin ham aday (box, kosinus) TOPLA - SAM kare basina 1 kez.
  3) Esikleri offline tara (modeli tekrar calistirmadan): her esikte P/R/F1 + genel AP@0.5.
  4) En iyi F1 esigini oner ve gorev3/calibrated.json'a yaz.

Kullanim:
  .venv\\Scripts\\python.exe -m gorev3.tools.calibrate --gt mini_val.json
  .venv\\Scripts\\python.exe -m gorev3.tools.calibrate --selftest
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BBox = tuple


# ----------------- saf metrik fonksiyonlari (self-test edilebilir) -----------------
def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate_at_threshold(rows, thr, iou_thr=0.5):
    """rows: her biri {present, gt_bbox, cand_box, score, area_frac}.
    Emit kurali: cand_box var & score>=thr & alan sinirlarinda (area_ok onceden isaretli).
    """
    tp = fp = fn = 0
    for r in rows:
        emit = (r["cand_box"] is not None) and (r["score"] >= thr) and r.get("area_ok", True)
        if r["present"]:
            if emit and iou(r["cand_box"], r["gt_bbox"]) >= iou_thr:
                tp += 1
            else:
                fn += 1           # nesne var ama tutturamadik (gonderilmedi veya IoU<0.5)
            if emit and iou(r["cand_box"], r["gt_bbox"]) < iou_thr:
                fp += 1           # ustelik yanlis yere kutu gonderdik
        else:
            if emit:
                fp += 1           # nesne yok ama kutu gonderdik
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def average_precision(rows, iou_thr=0.5):
    """Skora gore siralanmis AP@0.5 (esik-bagimsiz tavan; siralamanin kalitesi)."""
    dets = [r for r in rows if r["cand_box"] is not None and r.get("area_ok", True)]
    dets.sort(key=lambda r: r["score"], reverse=True)
    n_pos = sum(1 for r in rows if r["present"])
    if n_pos == 0:
        return 0.0
    tp = 0; fp = 0; prec_rec = []
    for r in dets:
        if r["present"] and iou(r["cand_box"], r["gt_bbox"]) >= iou_thr:
            tp += 1
        else:
            fp += 1
        prec_rec.append((tp / (tp + fp), tp / n_pos))
    # 11-nokta / monoton zarf ile alan
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        ps = [p for p, rc in prec_rec if rc >= t]
        ap += (max(ps) if ps else 0.0) / 101
    return ap


# ----------------- aday toplama (modeli kare basina 1 kez calistir) -----------------
def gather_candidates(gt_records, data_dir, cfg):
    from gorev3.reference_matcher import ReferenceMatcher
    import cv2, glob

    rm = ReferenceMatcher(cfg)
    # referanslari kaydet (video adina gore dogru klasor)
    ref_dirs = {
        "THYZ_2026_Ornek_Veri_1.MP4": "THYZ_2026_Ornek_Veri_1_Referans_Nesneler",
        "THYZ_2026_Ornek_Veri_2_Termal.MP4": "THYZ_2026_Ornek_Veri_2_Termal_Referans_Nesneler",
    }
    registered = set()
    def reg(video, ref):
        key = (video, ref)
        if key in registered:
            return
        p = os.path.join(data_dir, ref_dirs.get(video, ""), ref)
        rm.register_reference(f"{video}::{ref}", p)
        registered.add(key)

    by_frame = defaultdict(list)
    for r in gt_records:
        by_frame[(r["video"], r["frame_index"])].append(r)

    rows = []
    for (video, fidx), recs in sorted(by_frame.items()):
        cap = cv2.VideoCapture(os.path.join(data_dir, video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read(); cap.release()
        if not ok:
            print(f"UYARI: kare okunamadi {video}#{fidx}"); continue
        thermal = "termal" in video.lower()
        ctx = rm.new_frame(frame, thermal=thermal)
        fh, fw = frame.shape[:2]
        for rec in recs:
            reg(video, rec["reference"])
            box, score, tier = rm.candidate(ctx, f"{video}::{rec['reference']}")
            area_ok = True
            if box is not None:
                af = ((box[2]-box[0])*(box[3]-box[1]))/(fw*fh)
                area_ok = cfg.min_bbox_area_frac <= af <= cfg.max_bbox_area_frac
            rows.append(dict(
                video=video, frame=fidx, reference=rec["reference"],
                present=bool(rec["present"]),
                gt_bbox=tuple(rec["bbox"]) if rec.get("bbox") else None,
                cand_box=tuple(box) if box is not None else None,
                score=float(score), tier=tier, area_ok=area_ok,
            ))
    return rows


def sweep_and_report(rows, lo=0.30, hi=0.75, step=0.01):
    thrs = [round(lo + i * step, 4) for i in range(int((hi - lo) / step) + 1)]
    metrics = [evaluate_at_threshold(rows, t) for t in thrs]
    best = max(metrics, key=lambda m: (m["f1"], m["precision"]))
    ap = average_precision(rows)
    print(f"\nAP@0.5 (esik-bagimsiz tavan, siralamaya gore): {ap:.3f}")
    print(f"{'thr':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6}")
    for m in metrics:
        mark = "  <== EN IYI" if m is best else ""
        if abs((m['thr']*100) % 5) < 1e-6 or m is best:  # her 0.05'te bir + en iyi
            print(f"{m['thr']:.2f} {m['tp']:3d} {m['fp']:3d} {m['fn']:3d} "
                  f"{m['precision']:.3f} {m['recall']:.3f} {m['f1']:.3f}{mark}")
    print(f"\nONERI: samdino_min_cos = {best['thr']:.2f}  "
          f"(F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f})")
    return best, ap


def selftest():
    print("=== SELF-TEST (sentetik GT) ===")
    assert abs(iou((0,0,10,10),(0,0,10,10)) - 1.0) < 1e-9
    assert abs(iou((0,0,10,10),(5,0,15,10)) - (50/150)) < 1e-9
    assert iou((0,0,10,10),(20,20,30,30)) == 0.0
    rows = [
        # present, iyi kutu, yuksek skor -> yuksek esikte TP
        dict(present=True, gt_bbox=(0,0,10,10), cand_box=(0,0,10,10), score=0.70, area_ok=True),
        # present, iyi kutu, orta skor
        dict(present=True, gt_bbox=(0,0,10,10), cand_box=(1,1,11,11), score=0.50, area_ok=True),
        # absent, kutu var, dusuk skor -> yuksek esik bunu eler (FP onlenir)
        dict(present=False, gt_bbox=None, cand_box=(0,0,10,10), score=0.40, area_ok=True),
        # present, kutu yok -> her esikte FN
        dict(present=True, gt_bbox=(0,0,10,10), cand_box=None, score=0.0, area_ok=True),
    ]
    m_low = evaluate_at_threshold(rows, 0.35)
    m_high = evaluate_at_threshold(rows, 0.60)
    print("esik 0.35:", {k: m_low[k] for k in ("tp","fp","fn","precision","recall","f1")})
    print("esik 0.60:", {k: m_high[k] for k in ("tp","fp","fn","precision","recall","f1")})
    assert m_low["fp"] == 1 and m_high["fp"] == 0, "yuksek esik FP'yi elemeli"
    assert m_high["tp"] == 1, "0.60'ta yalnizca 0.70 skorlu TP kalir"
    ap = average_precision(rows)
    print(f"AP@0.5 = {ap:.3f}")
    print("SELF-TEST GECTI [OK]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", help="mini-val GT JSON yolu")
    ap.add_argument("--data", default=r"C:\Users\Acer\Desktop\drone\hyz\THYZ_2026_Ornek_Veri_Seti")
    ap.add_argument("--out", default=r"C:\Users\Acer\Desktop\drone\hyz\gorev3\calibrated.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.gt:
        print("--gt gerekli (veya --selftest). Once tools/label_minival.py ile etiketle."); return

    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from gorev3.config import Gorev3Config
    cfg = Gorev3Config()
    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    print(f"{len(gt)} GT kaydi; adaylar toplaniyor (SAM kare basina 1 kez)...")
    rows = gather_candidates(gt, args.data, cfg)
    color_rows = [r for r in rows if r["tier"] in ("compact", "scene")]
    gray_rows = [r for r in rows if r["tier"] == "gray"]
    out = {}

    print(f"\n### RENKLI referanslar ({len(color_rows)} satir) — samdino_min_cos ###")
    best_c, ap_c = sweep_and_report(color_rows, lo=0.30, hi=0.75)
    out.update(samdino_min_cos=best_c["thr"], ap_color=ap_c, f1_color=best_c["f1"],
               n_color=len(color_rows))

    if gray_rows:
        print(f"\n### GRI referanslar ({len(gray_rows)} satir) — samdino_min_cos_gray (CLAHE-gri) ###")
        best_g, ap_g = sweep_and_report(gray_rows, lo=0.15, hi=0.55)
        out.update(samdino_min_cos_gray=best_g["thr"], ap_gray=ap_g, f1_gray=best_g["f1"],
                   n_gray=len(gray_rows))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nyazildi: {args.out}  (config bunu okuyacak sekilde guncellenebilir)")


if __name__ == "__main__":
    main()
