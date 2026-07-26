"""Mini-dogrulama seti etiketleyici (kalibrasyon icin GT uretir).

Iki mod:
  --proposals : HEADLESS. Kareleri ornekler, sistemin ONERDIGI kutulari on-doldurur ve
                bir GT sablonu JSON yazar. GUI gerektirmez; kullanici JSON'u metin
                editorunde gozden gecirip duzeltir (present true/false, bbox koordinat).
  (varsayilan): GUI. OpenCV penceresinde her (kare,referans) icin oneri kutusu gosterilir;
                kullanici fare ile duzeltir. Tuslar: [s] oneriyi kabul  [a] cizili kutuyu
                kabul  [x] nesne yok  [n] sonraki  [p] onceki  [q] kaydet+cik.

Cikti semasi (calibrate.py bunu okur):
  [{"video","frame_index","reference","present":bool,"bbox":[x0,y0,x1,y1]|null}, ...]

Kullanim:
  .venv\\Scripts\\python.exe -m gorev3.tools.label_minival --proposals --out mini_val.json
  .venv\\Scripts\\python.exe -m gorev3.tools.label_minival --gt mini_val.json   # GUI ile duzelt
"""
from __future__ import annotations
import os, sys, json, glob, argparse

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DEFAULT = r"C:\Users\Acer\Desktop\drone\hyz\THYZ_2026_Ornek_Veri_Seti"
VIDEOS = {
    "THYZ_2026_Ornek_Veri_1.MP4": "THYZ_2026_Ornek_Veri_1_Referans_Nesneler",
    "THYZ_2026_Ornek_Veri_2_Termal.MP4": "THYZ_2026_Ornek_Veri_2_Termal_Referans_Nesneler",
}


def sample_indices(video_path, n):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return [int(total * (i + 1) / (n + 1)) for i in range(n)]


def build_matcher(data_dir, cfg):
    from gorev3.reference_matcher import ReferenceMatcher
    rm = ReferenceMatcher(cfg)
    refs_by_video = {}
    for video, refdir in VIDEOS.items():
        paths = sorted(glob.glob(os.path.join(data_dir, refdir, "Referans_Nesne_*")))
        refs_by_video[video] = paths
        for p in paths:
            rm.register_reference(f"{video}::{os.path.basename(p)}", p)
    return rm, refs_by_video


def gen_proposals(data_dir, out, n_per_video, cfg):
    """HEADLESS: her (kare,referans) icin sistem onerisini on-doldurulmus GT sablonu yaz."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import cv2
    rm, refs_by_video = build_matcher(data_dir, cfg)
    records = []
    for video, refpaths in refs_by_video.items():
        vpath = os.path.join(data_dir, video)
        thermal = "termal" in video.lower()
        for fidx in sample_indices(vpath, n_per_video):
            cap = cv2.VideoCapture(vpath); cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read(); cap.release()
            if not ok:
                continue
            ctx = rm.new_frame(frame, thermal=thermal)
            for p in refpaths:
                ref = os.path.basename(p)
                box, score, tier = rm.candidate(ctx, f"{video}::{ref}")
                records.append({
                    "video": video, "frame_index": fidx, "reference": ref,
                    # ONERI: sistem esigi asiyorsa present tahmin edilir; kullanici DUZELTMELI
                    "present": bool(box is not None and score >= cfg.samdino_min_cos),
                    "bbox": [round(float(v), 1) for v in box] if box is not None else None,
                    "_system_score": round(float(score), 3), "_tier": tier,
                    "_NOTE": "present ve bbox'i GERCEGE gore duzelt; _ ile baslayan alanlar yok sayilir",
                })
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    npos = sum(1 for r in records if r["present"])
    print(f"{len(records)} kayit yazildi ({npos} 'present' tahmini) -> {out}")
    print("SONRAKI: JSON'u ac, her kaydin present/bbox degerini gercege gore duzelt, "
          "sonra: python -m gorev3.tools.calibrate --gt " + out)


def run_gui(data_dir, gt_path, cfg):
    """Interaktif OpenCV etiketleyici. Kullanici yerelde `!` ile calistirir."""
    import cv2
    with open(gt_path, encoding="utf-8") as f:
        records = json.load(f)
    state = {"box": None, "drawing": False, "p0": None}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True; state["p0"] = (x, y); state["box"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x0, y0 = state["p0"]; state["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False

    cv2.namedWindow("label"); cv2.setMouseCallback("label", on_mouse)
    i = 0
    cache = {}
    while 0 <= i < len(records):
        r = records[i]
        vpath = os.path.join(data_dir, r["video"])
        key = (r["video"], r["frame_index"])
        if key not in cache:
            cap = cv2.VideoCapture(vpath); cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_index"])
            ok, fr = cap.read(); cap.release(); cache[key] = fr if ok else None
        frame = cache[key]
        if frame is None:
            i += 1; continue
        state["box"] = list(r["bbox"]) if r.get("bbox") else None
        while True:
            disp = frame.copy()
            # referans kucuk resmi kose
            rp = os.path.join(data_dir, VIDEOS[r["video"]], r["reference"])
            thumb = cv2.imread(rp)
            if thumb is not None:
                th = cv2.resize(thumb, (160, int(160 * thumb.shape[0] / thumb.shape[1])))
                disp[0:th.shape[0], 0:th.shape[1]] = th
            if state["box"]:
                x0, y0, x1, y1 = [int(v) for v in state["box"]]
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            txt = f"{i+1}/{len(records)} {r['reference']} present={r.get('present')} score={r.get('_system_score')}"
            cv2.putText(disp, txt, (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(disp, "[s]oneri [a]cizili [x]yok [n]ileri [p]geri [q]kaydet-cik",
                        (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("label", disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('n'), ord('a'), ord('s'), ord('x')):
                if k == ord('x'):
                    r["present"] = False; r["bbox"] = None
                elif k == ord('a') and state["box"]:
                    r["present"] = True; r["bbox"] = [round(float(v), 1) for v in state["box"]]
                elif k == ord('s'):
                    r["present"] = r.get("bbox") is not None
                i += 1; break
            elif k == ord('p'):
                i = max(0, i - 1); break
            elif k == ord('q'):
                cv2.destroyAllWindows()
                with open(gt_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                print(f"kaydedildi -> {gt_path}")
                return
    cv2.destroyAllWindows()
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"kaydedildi -> {gt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out", default="mini_val.json")
    ap.add_argument("--gt", help="GUI ile duzeltilecek mevcut GT JSON")
    ap.add_argument("--proposals", action="store_true", help="headless: oneri sablonu uret")
    ap.add_argument("--n", type=int, default=12, help="video basina kare sayisi")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from gorev3.config import Gorev3Config
    cfg = Gorev3Config()
    if args.proposals:
        gen_proposals(args.data, args.out, args.n, cfg)
    elif args.gt:
        run_gui(args.data, args.gt, cfg)
    else:
        print("--proposals (headless sablon) veya --gt <json> (GUI duzeltme) ver.")


if __name__ == "__main__":
    main()
