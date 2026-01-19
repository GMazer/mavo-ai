from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
print("Loading model...")
model = joblib.load('model_unified.pkl')

class SizeInput(BaseModel):
    cao: float
    nang: float
    nguc: float = None
    eo: float = None
    mong: float = None

# --- HÀM MỚI: HIỆU CHỈNH ĐỘ TIN CẬY ---
def adjust_confidence(raw_prob, data: SizeInput):
    """
    Hàm này giúp giảm độ tự tin của AI xuống mức thực tế
    nếu khách hàng cung cấp thiếu thông tin.
    """
    confidence = raw_prob
    
    # 1. Đếm số lượng thông tin bị thiếu
    missing_count = 0
    if data.nguc is None or data.nguc <= 0: missing_count += 1
    if data.eo is None or data.eo <= 0: missing_count += 1
    if data.mong is None or data.mong <= 0: missing_count += 1
    
    # 2. Áp dụng hình phạt (Penalty)
    # - Nếu thiếu 3 vòng (chỉ có Cao/Nặng): Trừ 10% - 15% độ tin cậy
    # - Nếu thiếu 1-2 vòng: Trừ 5% - 8%
    if missing_count == 3:
        confidence = confidence * 0.85 # Giảm 15%
    elif missing_count > 0:
        confidence = confidence * 0.92 # Giảm 8%
        
    # 3. Giới hạn trần (Cap)
    # Không bao giờ cho phép 100%, tối đa chỉ 98% cho "người"
    confidence = min(0.98, confidence)
    
    # 4. Giới hạn sàn (Floor)
    # Không để thấp quá gây hoang mang, tối thiểu 60%
    confidence = max(0.60, confidence)
    
    return confidence

@app.post("/predict")
def predict_size(data: SizeInput):
    # Xử lý input
    nguc_val = data.nguc if data.nguc is not None and data.nguc > 0 else -1
    eo_val   = data.eo   if data.eo   is not None and data.eo   > 0 else -1
    mong_val = data.mong if data.mong is not None and data.mong > 0 else -1
    hieu_so = data.cao - data.nang
    
    input_df = pd.DataFrame([{
        'Cao': data.cao, 'Can_nang': data.nang,
        'Nguc': nguc_val, 'Eo': eo_val, 'Mong': mong_val,
        'Hieu_So': hieu_so
    }])
    
    # Lấy xác suất gốc từ AI
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    
    # Tìm Top 1 và Top 2
    top_idx = np.argsort(probs)[-2:]
    size_1 = classes[top_idx[1]]
    size_2 = classes[top_idx[0]]
    raw_score_1 = probs[top_idx[1]]
    raw_score_2 = probs[top_idx[0]]
    
    # --- ÁP DỤNG HIỆU CHỈNH ---
    final_score = adjust_confidence(raw_score_1, data)
    
    # Logic trả lời
    method = "Phân tích đa chiều"
    if nguc_val > 0 and eo_val > 0 and mong_val > 0:
        method = "Chính xác cao (Đủ 5 chỉ số)"
    elif nguc_val > 0 or eo_val > 0 or mong_val > 0:
        method = "Kết hợp số đo & ước lượng"
    else:
        method = "Ước lượng theo Chiều cao/Cân nặng"

    # Tư vấn
    result = {}
    
    # Nếu chênh lệch giữa 2 size quá thấp (AI phân vân)
    if (raw_score_1 - raw_score_2) < 0.15:
        result = {
            "size": f"{size_1} hoặc {size_2}",
            "percent": f"{final_score:.0%}", # Trả về số % đã làm mượt
            "method": method,
            "message": f"Hệ thống phân vân giữa {size_2} và {size_1}. Bạn nên chọn theo sở thích (ôm/rộng)."
        }
    else:
        result = {
            "size": size_1,
            "percent": f"{final_score:.0%}", # Trả về số % đã làm mượt
            "method": method,
            "message": f"Size {size_1} là lựa chọn tối ưu nhất cho bạn."
        }
        
    return result
