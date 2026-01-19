from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import Optional

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
    # Thêm Optional[...] vào 3 dòng dưới để chấp nhận null
    nguc: Optional[float] = None
    eo: Optional[float] = None
    mong: Optional[float] = None

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
    # 1. XỬ LÝ DỮ LIỆU ĐẦU VÀO (Như cũ)
    nguc_val = data.nguc if data.nguc is not None and data.nguc > 0 else -1
    eo_val   = data.eo   if data.eo   is not None and data.eo   > 0 else -1
    mong_val = data.mong if data.mong is not None and data.mong > 0 else -1
    hieu_so = data.cao - data.nang
    
    input_df = pd.DataFrame([{
        'Cao': data.cao, 'Can_nang': data.nang,
        'Nguc': nguc_val, 'Eo': eo_val, 'Mong': mong_val,
        'Hieu_So': hieu_so
    }])
    
    # 2. DỰ ĐOÁN (Lấy Top 1 và Top 2)
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    
    # Sắp xếp xác suất tăng dần
    top_idx = np.argsort(probs)[-2:] 
    
    size_1 = classes[top_idx[1]] # Size tốt nhất (Top 1)
    size_2 = classes[top_idx[0]] # Size tốt nhì (Top 2)
    
    raw_score_1 = probs[top_idx[1]]
    raw_score_2 = probs[top_idx[0]]
    
    # 3. TÍNH ĐỘ TIN CẬY (Áp dụng hình phạt nếu thiếu thông tin)
    final_score = adjust_confidence(raw_score_1, data)
    
    # Xác định phương pháp dự đoán (để hiển thị)
    method = "Phân tích đa chiều"
    if nguc_val > 0 and eo_val > 0 and mong_val > 0:
        method = "Chính xác cao (Full chỉ số)"
    elif nguc_val > 0 or eo_val > 0 or mong_val > 0:
        method = "Kết hợp số đo & ước lượng"
    else:
        method = "Ước lượng theo Chiều cao/Cân nặng"

    # 4. LOGIC TRẢ KẾT QUẢ (Đã nâng cấp)
    result = {}
    
    # TRƯỜNG HỢP A: AI thực sự phân vân (Tỷ lệ bầu chọn ngang ngửa nhau)
    # Ví dụ: Size M (48%) vs Size L (45%) -> Chênh lệch < 15%
    if (raw_score_1 - raw_score_2) < 0.15:
        result = {
            "size": f"{size_1} hoặc {size_2}",
            "percent": f"{final_score:.0%}",
            "method": method,
            "message": f"Hệ thống phân vân giữa {size_2} và {size_1}. Bạn nên chọn theo sở thích (thích rộng lấy {size_1}, thích ôm lấy {size_2})."
        }
        
    # TRƯỜNG HỢP B: AI chọn được size, nhưng độ tin cậy thấp (< 80%)
    # Ví dụ: Chỉ nhập Cao/Nặng, hoặc số đo lạ -> Tin cậy 70%
    elif final_score < 0.80:
        result = {
            # Hiển thị kiểu: "M (Nên thử thêm L)"
            "size": f"{size_1}", 
            "percent": f"{final_score:.0%}",
            "method": method,
            # Gợi ý thêm Size 2 trong lời nhắn
            "message": f"⚠️ Độ tin cậy thấp (<80%) do thiếu thông tin hoặc số đo lạ. AI khuyên bạn chọn **{size_1}**, nhưng hãy cân nhắc thử thêm **{size_2}** cho chắc chắn."
        }
        
    # TRƯỜNG HỢP C: Tin cậy cao (>= 80%) -> Chốt đơn 1 size
    else:
        result = {
            "size": size_1,
            "percent": f"{final_score:.0%}",
            "method": method,
            "message": f"Tuyệt vời! Dựa trên số liệu, Size {size_1} là lựa chọn chuẩn xác nhất cho bạn."
        }
        
    return result