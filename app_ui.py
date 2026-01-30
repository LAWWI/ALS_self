import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

# ตั้งค่า Path ให้ตรงกับ Docker Volume
BASE_PATH = "/app"
RESULT_PATH = os.path.join(BASE_PATH, "output/final_recommend")
MOVIE_LIST_PATH = os.path.join(BASE_PATH, "output/movie_list")
MODEL_FACTORS_PATH = os.path.join(BASE_PATH, "model/als_model/itemFactors")

def load_csv(path):
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if files: return pd.read_csv(os.path.join(path, files[0]))
    return None

def load_factors(path):
    if os.path.exists(path):
        try: return pd.read_parquet(path, engine='pyarrow')
        except: return None
    return None

df_recs = load_csv(RESULT_PATH)
df_movies = load_csv(MOVIE_LIST_PATH)
df_factors = load_factors(MODEL_FACTORS_PATH)

st.title("🎬 AI Movie Recommender (Hybrid System)")

if df_movies is not None:
    # ส่วนที่ 1: Batch Recommend
    st.subheader("🔍 แนะนำรายบุคคล (Batch)")
    if df_recs is not None:
        user_id = st.selectbox("เลือก User ID:", sorted(df_recs['userId'].unique()))
        if st.button("แสดงคำแนะนำ"):
            st.success(df_recs[df_recs['userId'] == user_id]['recommended_movies'].values[0])

    st.divider()

    # ส่วนที่ 2: Similarity Calculation
    st.subheader("🎞️ ค้นหาหนังที่ใกล้เคียง (Real-time AI)")
    movie_choice = st.selectbox("เลือกหนังที่คุณชอบ:", df_movies['title'].sort_values().unique())
    
    if movie_choice:
        target_id = df_movies[df_movies['title'] == movie_choice]['movieId'].values[0]
        
        if df_factors is not None:
            target_row = df_factors[df_factors['id'] == target_id]
            if not target_row.empty:
                # คำนวณความคล้าย
                target_v = np.array(target_row['features'].values[0]).reshape(1, -1)
                all_vectors = np.array([np.array(x) for x in df_factors['features'].values])
                all_ids = df_factors['id'].values
                
                # 
                scores = cosine_similarity(target_v, all_vectors)[0]
                top_idx = scores.argsort()[-6:-1][::-1] # เอา 5 อันดับแรกที่ไม่ใช่ตัวเอง
                
                st.write("### 🤖 หนังที่ AI แนะนำจริงๆ:")
                for i in top_idx:
                    sim_id = all_ids[i]
                    m_title = df_movies[df_movies['movieId'] == sim_id]['title'].values[0]
                    st.write(f"⭐ **{m_title}** (ความคล้าย: {scores[i]:.2%})")
            else:
                st.warning("หนังเรื่องนี้ยังไม่มีข้อมูลใน Model AI")
else:
    st.error("ไม่พบข้อมูล กรุณารัน Spark ให้เสร็จก่อน")