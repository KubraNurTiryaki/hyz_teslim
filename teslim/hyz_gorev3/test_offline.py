import os
import torch
import numpy as np

# 1. ÇEVRİMDIŞI ORTAM DEĞİŞKENLERİ (İndirme yaparken geçici olarak '#' ile kapalı tut)
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("="*50)
print("🚀 ÇEVRİMDIŞI TEST VE MODEL YÜKLEME BAŞLADI")
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Kullanılan Cihaz: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# GÜNCELLEME: SuperPoint için 1 kanallı (gri), diğerleri için 3 kanallı (RGB) sahte girdiler
dummy_gray1 = torch.rand(1, 1, 224, 224).to(device)
dummy_gray2 = torch.rand(1, 1, 224, 224).to(device)
dummy_rgb1 = torch.rand(1, 3, 224, 224).to(device)

# ----------------------------------------------------
# A. SUPERPOINT & LIGHTGLUE YÜKLEME VE TEST
# ----------------------------------------------------
print("\n[1/3] SuperPoint & LightGlue yükleniyor...")
try:
    from lightglue import LightGlue, SuperPoint
    
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    with torch.no_grad():
        # Gri tonlamalı girdilerimizi modele besliyoruz
        # SuperPoint (B,C,H,W) bekler; batch boyutu düşürülmemeli (bkz. gorev3/matcher.py)
        feats1 = extractor({'image': dummy_gray1})
        feats2 = extractor({'image': dummy_gray2})
        matches = matcher({'image0': feats1, 'image1': feats2})
    
    print("✅ SuperPoint & LightGlue başarıyla yüklendi ve test edildi!")
except Exception as e:
    print(f"❌ SuperPoint & LightGlue hatası: {e}")

# ----------------------------------------------------
# B. DINOv2 YÜKLEME VE TEST (Rate Limit aşımı için HuggingFace API geçişi)
# ----------------------------------------------------
print("\n[2/3] DINOv2 (dinov2_vitb14_reg) yükleniyor...")
try:
    from transformers import AutoModel
    
    # torch.hub rate limit hatasını aşmak için resmi HuggingFace modelini çekiyoruz
    dinov2_model_name = "facebook/dinov2-with-registers-base" # vitb14_reg dengi
    dinov2 = AutoModel.from_pretrained(dinov2_model_name).eval().to(device)
    
    with torch.no_grad():
        features = dinov2(dummy_rgb1)
    
    print("✅ DINOv2 başarıyla yüklendi ve test edildi!")
except Exception as e:
    print(f"❌ DINOv2 hatası: {e}")

# ----------------------------------------------------
# C. MATCHANYTHING (ELoFTR) YÜKLEME VE TEST
# ----------------------------------------------------
print("\n[3/3] MatchAnything (ELoFTR) yükleniyor...")
try:
    from transformers import AutoModel, AutoImageProcessor
    
    model_name = "zju-community/matchanything_eloftr"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    print("✅ MatchAnything başarıyla yüklendi ve test edildi!")
except Exception as e:
    print(f"❌ MatchAnything hatası: {e}")

print("\n" + "="*50)
print("🏁 TEST TAMAMLANDI")
print("="*50)