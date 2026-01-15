from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Cấu hình CORS (Để web Firebase gọi được API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model_basic = joblib.load('model_basic.pkl')
model_full = joblib.load('model_full.pkl')

class SizeInput(BaseModel):
    cao: float
    nang: float
    nguc: float = None
    eo: float = None
    mong: float = None

@app.get("/")
def home():
    return {"status": "Mavo AI is ready!"}

@app.post("/predict")
def predict_size(data: SizeInput):
    hieu_so = data.cao - data.nang
    
    # TRƯỜNG HỢP 1: ĐỦ 3 VÒNG -> MODEL FULL
    if data.nguc and data.eo and data.mong:
        input_df = pd.DataFrame([{
            'Cao': data.cao, 'Can_nang': data.nang,
            'Nguc': data.nguc, 'Eo': data.eo, 'Mong': data.mong,
            'Hieu_So': hieu_so
        }])
        pred = model_full.predict(input_df)[0]
        return {
            "size": pred,
            "method": "Chính xác (5 chỉ số)",
            "message": f"Dựa trên số đo 3 vòng, size {pred} là chuẩn nhất."
        }
        
    # TRƯỜNG HỢP 2: THIẾU SỐ ĐO -> MODEL BASIC
    else:
        input_df = pd.DataFrame([{'Cao': data.cao, 'Can_nang': data.nang, 'Hieu_So': hieu_so}])
        probs = model_basic.predict_proba(input_df)[0]
        classes = model_basic.classes_
        
        # Lấy 2 size cao điểm nhất
        import numpy as np
        top_idx = np.argsort(probs)[-2:]
        size_1 = classes[top_idx[1]]
        size_2 = classes[top_idx[0]]
        score_1 = probs[top_idx[1]]
        score_2 = probs[top_idx[0]]
        
        # Tư vấn
        if (score_1 - score_2) < 0.15:
            return {
                "size": f"{size_1} hoặc {size_2}",
                "method": "Cơ bản (Linh hoạt)",
                "message": f"Bạn ở ngưỡng giữa size {size_2} và {size_1}. Thích ôm chọn {size_2}, thích rộng chọn {size_1}."
            }
        else:
            return {
                "size": size_1,
                "method": "Cơ bản",
                "message": f"Dựa trên chiều cao cân nặng, size {size_1} là phù hợp."
            }