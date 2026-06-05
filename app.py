import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Bài Tập Lớn - Phát hiện gian lận", layout="wide")

# --- ĐIỀU HƯỚNG MENU SIDEBAR ---
st.sidebar.title("📁 Danh Mục Bài Tập Lớn")
chuyen_muc = st.sidebar.radio("Lựa chọn chức năng:", [
    "🧮 1. Các bài toán Python cơ bản (Mục 3)", 
    "🛡️ 2. Hệ thống phát hiện gian lận (Mục 1, 2, 4, 5)"
])

# ========================================================
# CHỨC NĂNG 1: CÁC BÀI TOÁN CƠ BẢN (MỤC 3 ĐỀ BÀI)
# ========================================================
if chuyen_muc == "🧮 1. Các bài toán Python cơ bản (Mục 3)":
    st.title("🧮 Thực Hành Các Bài Toán Python Cơ Bản")
    st.markdown("---")
    
    st.subheader("1️⃣ Tính giai thừa của n")
    n = st.number_input("Nhập số nguyên n (n >= 0):", min_value=0, value=5, step=1)
    if st.button("Tính Giai Thừa"):
        import math
        st.success(f"Kết quả: {n}! = {math.factorial(n)}")
        
    st.markdown("---")
    st.subheader("2️⃣ Tính giá trị trung bình của dãy số")
    day_so_str = st.text_input("Nhập dãy số (cách nhau bởi dấu phẩy):", "10, 20, 30, 40, 50")
    if st.button("Tính Trung Bình"):
        try:
            day_so = [float(x.strip()) for x in day_so_str.split(",")]
            st.success(f"Giá trị trung bình của dãy số là: {sum(day_so) / len(day_so)}")
        except:
            st.error("Vui lòng nhập đúng định dạng số, ví dụ: 10, 20, 30")

    st.markdown("---")
    st.subheader("3️⃣ Tính lợi nhuận tích lũy sau 12 tháng")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        goc = st.number_input("Số tiền gốc ban đầu (VND):", min_value=0.0, value=10000000.0, step=500000.0)
    with col_t2:
        lai_suat = st.number_input("Tỷ lệ lãi suất năm (%):", min_value=0.0, value=6.0, step=0.1)
        
    if st.button("Tính Lợi Nhuận"):
        lai_suat_thang = (lai_suat / 100) / 12
        tien_hien_tai = goc
        for _ in range(12):
            tien_hien_tai += tien_hien_tai * lai_suat_thang
        loi_nhuan = tien_hien_tai - goc
        st.success(f"Lợi nhuận thu được sau 12 tháng: {loi_nhuan:,.0f} VND (Tổng tiền nhận về: {tien_hien_tai:,.0f} VND)")

# ========================================================
# CHỨC NĂNG 2: PHÁT HIỆN GIAN LẬN (MỤC 1, 2, 4, 5 ĐỀ BÀI)
# ========================================================
else:
    st.title("🛡️ Hệ Thống Phân Tích & Phát Hiện Giao Dịch Bất Thường")
    st.markdown("---")

    # ĐỔI TÊN HÀM Ở ĐÂY ĐỂ ÉP STREAMLIT XÓA CACHE CŨ LẬP TỨC
    @st.cache_resource
    def he_thong_ai_hoc_du_lieu():
        file_data = 'financial_anomaly_data.csv'
        
        if os.path.exists(file_data):
            df = pd.read_csv(file_data).copy()
            
            # Tìm cột nhãn
            target_col = None
            for c in df.columns:
                if c.lower() in ['is_anomaly', 'anomaly', 'fraud', 'class', 'label']:
                    target_col = c
                    break
            if target_col is None:
                target_col = df.columns[-1]
            
            # Khởi tạo danh sách cột số an toàn
            cac_cot_so = []
            
            # Duyệt qua các cột để ép kiểu chuỗi chữ thành số bằng LabelEncoder
            for cot in df.columns:
                if pd.api.types.is_numeric_dtype(df[cot]):
                    cac_cot_so.append(cot)
                elif pd.api.types.is_string_dtype(df[cot]) or pd.api.types.is_object_dtype(df[cot]):
                    if df[cot].nunique() <= 50 or cot == target_col:
                        le = LabelEncoder()
                        df[cot] = le.fit_transform(df[cot].astype(str))
                        cac_cot_so.append(cot)
            
            df_clean = df[cac_cot_so].dropna()
            
            X = df_clean.drop(target_col, axis=1)
            y = df_clean[target_col].astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            return model, cm, X.columns.tolist()
                
        return None, None, None

    model, cm, feature_columns = he_thong_ai_hoc_du_lieu()

    if model is None:
        st.error("❌ Không tìm thấy file dữ liệu 'financial_anomaly_data.csv' tại Desktop.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("✍️ Nhập thông tin giao dịch cần kiểm tra")
            user_inputs = {}
            for col_name in feature_columns:
                default_val = 500000.0 if 'amount' in col_name.lower() else 0.0
                user_inputs[col_name] = st.number_input(f"Nhập thông số [{col_name}]:", min_value=0.0, value=default_val)
                
            nut_phan_tich = st.button("🚀 Kích Hoạt AI Phân Tích")

        with col2:
            st.subheader("📊 Kết quả phân tích & Đánh giá")
            if nut_phan_tich:
                du_lieu_dau_vao = np.array([[user_inputs[col] for col in feature_columns]])
                du_doan = model.predict(du_lieu_dau_vao)
                xac_suat = model.predict_proba(du_lieu_dau_vao)
                
                if du_doan[0] == 1:
                    st.error("🚨 CẢNH BÁO: Giao dịch này có dấu hiệu GIAN LẬN / BẤT THƯỜNG!")
                    st.metric(label="Tỷ lệ rủi ro gian lận", value=f"{xac_suat[0][1]*100:.2f}%")
                else:
                    st.success("✅ AN TOÀN: Giao dịch được đánh giá là BÌNH THƯỜNG.")
                    st.metric(label="Mức độ an toàn", value=f"{xac_suat[0][0]*100:.2f}%")
            else:
                st.info("Vui lòng nhập thông tin ở cột bên trái và bấm nút 'Kích Hoạt AI Phân Tích'.")
                
            with st.expander("🔍 Xem độ chính xác của Mô hình AI trên tập dữ liệu kiểm thử"):
                st.write("Ma trận nhầm lẫn (Confusion Matrix):")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
