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

print("Loading model...")
model = joblib.load('model_unified.pkl')

# --- 1. ĐỊNH NGHĨA LẠI BẢNG SIZE CHUẨN ĐỂ SO SÁNH ---
# Chúng ta dùng bảng này để "bắt lỗi" AI
SIZE_SPECS = {
    'S':  {'nguc_max': 84, 'eo_max': 68, 'mong_max': 90},
    'M':  {'nguc_max': 88, 'eo_max': 72, 'mong_max': 94},
    'L':  {'nguc_max': 92, 'eo_max': 76, 'mong_max': 98},
    'XL': {'nguc_max': 100, 'eo_max': 82, 'mong_max': 105} 
}
# Thứ tự size để so sánh lớn nhỏ
SIZE_ORDER = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
ORDER_TO_SIZE = {1: 'S', 2: 'M', 3: 'L', 4: 'XL'}

class SizeInput(BaseModel):
    cao: float
    nang: float
    nguc: Optional[float] = None
    eo: Optional[float] = None
    mong: Optional[float] = None

def adjust_confidence(raw_prob, data: SizeInput):
    confidence = raw_prob
    missing_count = 0
    if data.nguc is None or data.nguc <= 0: missing_count += 1
    if data.eo is None or data.eo <= 0: missing_count += 1
    if data.mong is None or data.mong <= 0: missing_count += 1
    
    if missing_count == 3: confidence *= 0.85 
    elif missing_count > 0: confidence *= 0.92
        
    confidence = min(0.98, confidence)
    confidence = max(0.60, confidence)
    return confidence

@app.post("/predict")
def predict_size(data: SizeInput):
    # 1. Xử lý Input
    nguc_val = data.nguc if data.nguc is not None and data.nguc > 0 else -1
    eo_val   = data.eo   if data.eo   is not None and data.eo   > 0 else -1
    mong_val = data.mong if data.mong is not None and data.mong > 0 else -1
    hieu_so = data.cao - data.nang
    
    input_df = pd.DataFrame([{
        'Cao': data.cao, 'Can_nang': data.nang,
        'Nguc': nguc_val, 'Eo': eo_val, 'Mong': mong_val,
        'Hieu_So': hieu_so
    }])
    
    # 2. AI Dự đoán (Lấy kết quả thô từ Model)
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[-2:]
    
    ai_size_1 = classes[top_idx[1]] # AI chọn cái này
    ai_size_2 = classes[top_idx[0]] # AI phân vân cái này
    raw_score = probs[top_idx[1]]
    
    # 3. --- LOGIC "PHỦ QUYẾT" (QUAN TRỌNG NHẤT) ---
    final_size = ai_size_1
    upgrade_msg = ""
    forced_upgrade = False
    
    # Kiểm tra xem size AI chọn có chứa nổi số đo khách không
    # Nếu khách nhập Ngực/Eo/Mông, ta so sánh với bảng SIZE_SPECS
    
    suggested_min_size_idx = 0
    reason_part = ""

    # Check Ngực
    if nguc_val > 0:
        for size, specs in SIZE_SPECS.items():
            if nguc_val <= specs['nguc_max']: # Tìm size nhỏ nhất vừa ngực
                break
            suggested_min_size_idx = max(suggested_min_size_idx, SIZE_ORDER.get(size, 0) + 1)
            if suggested_min_size_idx > 4: suggested_min_size_idx = 4 # Max là XL
            if suggested_min_size_idx > SIZE_ORDER.get(final_size, 0):
                reason_part = "vòng 1"

    # Check Eo (Nếu Ngực chưa đẩy size lên thì check tiếp Eo)
    if eo_val > 0:
        temp_idx = 0
        for size, specs in SIZE_SPECS.items():
            if eo_val <= specs['eo_max']:
                break
            temp_idx = SIZE_ORDER.get(size, 0) + 1
        
        if temp_idx > suggested_min_size_idx:
            suggested_min_size_idx = temp_idx
            if suggested_min_size_idx > 4: suggested_min_size_idx = 4
            reason_part = "vòng 2"

    # 4. SO SÁNH & CHỐT SIZE
    # Lấy chỉ số size AI đang chọn (Ví dụ M -> 2)
    current_ai_idx = SIZE_ORDER.get(ai_size_1, 1)
    
    # Nếu Size theo số đo > Size AI chọn => ÉP LẤY SIZE TO
    if suggested_min_size_idx > current_ai_idx:
        forced_upgrade = True
        final_size = ORDER_TO_SIZE[suggested_min_size_idx]
        # Nếu bị ép đổi size, size phụ sẽ là size AI chọn ban đầu
        size_2 = ai_size_1 
        upgrade_msg = f"Dựa trên chiều cao cân nặng thì size {ai_size_1} là vừa, TUY NHIÊN do {reason_part} của bạn cần sự thoải mái nên hệ thống ĐỀ XUẤT TĂNG LÊN size {final_size}."
    else:
        # Nếu AI chọn đúng hoặc chọn to hơn số đo thì giữ nguyên
        final_size = ai_size_1
        size_2 = ai_size_2

    # 5. Tính độ tin cậy & Output
    final_score = adjust_confidence(raw_score, data)
    
    # Xây dựng câu thông báo
    result = {}
    method = "Phân tích đa chiều"
    
    if forced_upgrade:
        result = {
            "size": final_size,
            "percent": "90%", # Tự tin cao vì dựa trên số đo thật
            "method": "Ưu tiên theo số đo vòng cơ thể",
            "message": f"🎯 {upgrade_msg} Hãy chọn **{final_size}** để mặc vừa vặn và thoải mái nhất nhé!"
        }
    else:
        # Logic cũ
        if (raw_score - probs[top_idx[0]]) < 0.15:
            result = {
                "size": f"{final_size} hoặc {size_2}",
                "percent": f"{final_score:.0%}",
                "method": method,
                "message": f"Hệ thống phân vân giữa {size_2} và {final_size}. Bạn nên cân nhắc chọn {final_size} cho rộng rãi."
            }
        elif final_score < 0.80:
            result = {
                "size": final_size, 
                "percent": f"{final_score:.0%}",
                "method": method,
                "message": f"AI khuyên bạn chọn **{final_size}**, nhưng hãy cân nhắc thử thêm **{size_2}** cho chắc chắn."
            }
        else:
            result = {
                "size": final_size,
                "percent": f"{final_score:.0%}",
                "method": method,
                "message": f"Tuyệt vời! Size {final_size} là lựa chọn tối ưu nhất cho số đo của bạn."
            }
            
    return result