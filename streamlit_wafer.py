# -*- coding: utf-8 -*-
# 반도체 웨이퍼 결함 패턴 분류 — Streamlit 데모 앱
# 실행: streamlit run streamlit_wafer.py  (또는: python -m streamlit run streamlit_wafer.py)
# 사전 준비: 노트북에서 학습한 model/wafer_cnn_model.h5, model/wafer_label_encoder.pkl 필요
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from matplotlib import cm
 
IMG_SIZE = 128
st.set_page_config(page_title="웨이퍼 결함 패턴 분류", layout="wide")
 
 
# ============================================================
# 공통: 패턴별 예시 웨이퍼맵 생성 (데모 + 클래스 정보 탭에서 공용 사용)
# 값 규칙: 0=웨이퍼 밖, 1=정상 die, 2=불량 die
# ============================================================
def make_pattern(name, seed=7):
    rng = np.random.RandomState(seed)
    s = IMG_SIZE
    yy, xx = np.ogrid[:s, :s]
    c = s / 2
    inside = (yy - c) ** 2 + (xx - c) ** 2 <= (c - 1) ** 2
    m = np.zeros((s, s)); m[inside] = 1
    d = (yy - c) ** 2 + (xx - c) ** 2
    # 약한 배경 노이즈 (실제 웨이퍼맵 느낌)
    m[(rng.rand(s, s) < 0.02) & inside] = 2
    if name == 'Center':
        m[(d <= (s * 0.20) ** 2) & inside] = 2
    elif name == 'Donut':
        m[(d <= (s * 0.40) ** 2) & (d >= (s * 0.27) ** 2) & inside] = 2
    elif name == 'Edge-Ring':
        m[(d >= (s * 0.40) ** 2) & inside] = 2
    elif name == 'Edge-Loc':
        ang = np.arctan2(yy - c, xx - c)
        m[(d >= (s * 0.38) ** 2) & (ang > 0.3) & (ang < 1.5) & inside] = 2
    elif name == 'Loc':
        cx, cy = 42, 20
        m[((yy - cy) ** 2 + (xx - cx) ** 2 <= (s * 0.13) ** 2) & inside] = 2
    elif name == 'Scratch':
        for x in range(s):
            yv = int(0.7 * x - 5)
            if 0 <= yv < s and inside[yv, x]:
                m[yv, x] = 2
    elif name == 'Random':
        m[(rng.rand(s, s) < 0.28) & inside] = 2
    elif name == 'Near-full':
        m[inside] = 2
        m[(rng.rand(s, s) < 0.08) & inside] = 1
    # 'none' = 배경 노이즈만
    return m
 
 
def thumb(name):
    """패턴 썸네일 figure 반환"""
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    ax.imshow(make_pattern(name), cmap='viridis')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    return fig
 
 
# ============================================================
# 결함 패턴 정보: (어떻게 나타나는지) + (추정 공정 원인 / 점검 포인트)
# ============================================================
PATTERN_INFO = {
    'Center':    ("웨이퍼 한가운데에 불량이 원형으로 뭉쳐 나타남",
                  "척(chuck) 중심부 온도·압력 편차, 스핀 코팅 중심 불균일, 중심부 가스 흐름 정체 의심"),
    'Donut':     ("중심은 비우고 중간 반경에 고리(도넛) 모양으로 나타남",
                  "반경 방향 공정 편차 — 가스 흐름·플라즈마 분포가 특정 반경대에서 불균일할 때 의심"),
    'Edge-Ring': ("가장자리 전체를 따라 링처럼 빙 둘러 나타남",
                  "에지 척킹·클램프 접촉 이슈, 에지 비드 제거 불량, 가장자리 식각·증착 균일도 저하 의심"),
    'Edge-Loc':  ("가장자리 중 일부 구간에만 호(arc) 형태로 나타남",
                  "에지 핸들링 중 특정 위치 접촉 손상, 클램프/핀 자국, 국부 에지 세정 불량 의심"),
    'Loc':       ("위치와 무관하게 한 구역에 작게 뭉쳐 나타남",
                  "파티클 낙하, 국부적 디펙트 소스 존재 의심 (가장자리에 국한되지 않음)"),
    'Scratch':   ("가늘고 길게 선(줄) 형태로 이어져 나타남",
                  "핸들링·이송 중 로봇 암/캐리어 기계적 긁힘, CMP·세정 중 스크래치 의심"),
    'Random':    ("전면에 규칙 없이 흩뿌려져 나타남",
                  "전반적 파티클 오염, 클린룸·장비 청정도 저하, 재료 자체 결함 의심"),
    'Near-full': ("웨이퍼 거의 전체가 불량으로 덮여 나타남 (정상이 드묾)",
                  "심각한 공정/장비 이상 또는 레시피 오류 — 즉시 장비 점검 필요"),
    'none':      ("뚜렷한 패턴 없이 산발적 점만 있음",
                  "정상 또는 산발적 불량 — 특이 공정 이슈 신호 아님"),
}
 
 
 
# ============================================================
# Grad-CAM — 결함 판단 근거 시각화 (XAI)
# ============================================================
def build_grad_model(_model, last_conv_name='last_conv'):
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = inp; conv_out = None
    for layer in _model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            conv_out = x
    return tf.keras.Model(inp, [conv_out, x])
 
def make_gradcam(img2d, _model, last_conv_name='last_conv', pred_index=None):
    arr = img2d[np.newaxis, ..., np.newaxis].astype('float32')
    grad_model = build_grad_model(_model, last_conv_name)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
 
def overlay_gradcam(wafer2d, heatmap, alpha=0.5):
    base = (cm.viridis(wafer2d / 2.0)[:, :, :3] * 255).astype('uint8')
    hm = cv2.resize(heatmap, (wafer2d.shape[1], wafer2d.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.clip(base * (1 - alpha) + hm * alpha, 0, 255).astype('uint8')
 
# ---------- 모델 로드 (캐싱) ----------
@st.cache_resource
def load_assets():
    model = load_model('model/wafer_cnn_model.h5')
    le = joblib.load('model/wafer_label_encoder.pkl')
    return model, le
 
try:
    model, le = load_assets()
    model_ready = True
except Exception:
    model_ready = False
 
 
st.title("🔬 반도체 웨이퍼 결함 패턴 분류 (CNN)")
st.caption("WM-811K 학습 모델 기반 · 웨이퍼맵 패턴을 진단하고 추정 공정 원인을 제시합니다")
if not model_ready:
    st.warning("학습된 모델을 찾을 수 없습니다. 먼저 노트북을 실행해 model/ 폴더에 모델을 저장하세요.")
 
tab1, tab2, tab3, tab4 = st.tabs(["📌 프로젝트 개요", "🖼️ 웨이퍼맵 예측", "📊 클래스 정보", "ℹ️ 모델 정보"])
 
# ============================================================
# 탭1 — 프로젝트 개요 (제조혁신 연계)
# ============================================================
with tab1:
    st.subheader("AI 기반 반도체 웨이퍼 결함 패턴 분류 모델 개발 및 시각화")
    st.markdown(
        "양산 FAB의 웨이퍼맵에 나타나는 **불량 die의 공간 패턴**을 CNN으로 자동 분류하고, "
        "패턴을 **공정 이상 원인으로 해석**하여 수율·품질 대응을 가속하는 Smart Manufacturing 적용 사례입니다."
    )
 
    c1, c2, c3 = st.columns(3)
    c1.metric("대상 데이터", "WM-811K", "실제 FAB 웨이퍼맵")
    c2.metric("결함 패턴", "8종 + 정상", "공간 패턴 분류")
    c3.metric("배포", "Streamlit", "현업 활용 UI")
 
    st.markdown("#### 추진 배경")
    st.markdown(
        "- **수율 직결 문제** — 웨이퍼맵의 결함 패턴은 특정 공정 이상(중심 온도 편차, 에지 척킹, 파티클 등)과 직결되어 수율·품질의 핵심 단서가 됩니다.\n"
        "- **기존 판독의 한계** — 엔지니어 육안 판독은 시간 소요·주관 개입·대량 처리에 한계가 있습니다.\n"
        "- **AI Transformation 필요** — CNN 자동 분류로 패턴을 즉시 식별하고 근본 원인 대응을 앞당깁니다."
    )
 
    st.markdown("#### 제조혁신 연계")
    j1, j2 = st.columns(2)
    with j1:
        st.markdown(
            "**Smart Manufacturing 구현 사례**\n\n"
            "단위 공정 데이터(웨이퍼맵)에 AI/ML을 적용해 양산 효율을 높이는 구체적 현장 적용 사례입니다."
        )
        st.markdown(
            "**IT × OT 융합 관점**\n\n"
            "OT(반도체 공정·장비) 도메인 지식과 IT(딥러닝/CNN)를 결합 — 패턴을 공정 원인으로 번역합니다."
        )
    with j2:
        st.markdown(
            "**Work With AI (업무 프로세스 혁신)**\n\n"
            "엔지니어 육안 검사를 AI 자동 분류로 전환해 반복 업무를 제거하고 판단을 지원합니다."
        )
        st.markdown(
            "**Digital Twin · 선도 기술 확장성**\n\n"
            "FDC/MES 연계 실시간 이상탐지, Digital Twin 기반 가상 검증으로 확장 가능한 구조입니다."
        )
 
    st.markdown("#### 기대 효과")
    st.markdown(
        "- 수율 이상의 **근본 원인 규명 시간 단축** (패턴 → 공정 원인 자동 매핑)\n"
        "- 결함 판독 **자동화·표준화**로 엔지니어 리소스 절감\n"
        "- 데이터 기반 의사결정 문화 및 **현장 AIX 체계**로의 발판"
    )
 
# ============================================================
# 탭2 — 웨이퍼맵 예측
# ============================================================
with tab2:
    st.subheader("웨이퍼맵 결함 패턴 예측")
    st.markdown("`.npy` 웨이퍼맵(2D 배열, 0/1/2 값)을 업로드하거나, 샘플 패턴을 생성해 테스트하세요.")
 
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("웨이퍼맵 업로드 (.npy)", type=['npy'])
    with col2:
        demo = st.selectbox("또는 샘플 패턴 생성",
                            ['(선택 안 함)'] + list(PATTERN_INFO.keys()))
 
    wafer = None
    if uploaded is not None:
        wafer = np.load(uploaded)
    elif demo != '(선택 안 함)':
        wafer = make_pattern(demo, seed=np.random.randint(0, 9999))
 
    if wafer is not None and model_ready:
        arr = cv2.resize(np.asarray(wafer, dtype='float32'), (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        proba = model.predict(arr[np.newaxis, ..., np.newaxis], verbose=0)[0]
        pred_cls = le.classes_[proba.argmax()]
 
        r1, r2 = st.columns([1, 1])
        with r1:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.imshow(arr, cmap='viridis'); ax.axis('off')
            ax.set_title(f'Pred: {pred_cls} ({proba.max():.1%})', fontsize=11)
            st.pyplot(fig, use_container_width=False)
        with r2:
            st.metric("예측 결함 패턴", pred_cls, f"{proba.max():.1%}")
            appear, cause = PATTERN_INFO.get(pred_cls, ("-", "-"))
            st.info(f"**나타나는 형태:** {appear}\n\n**추정 공정 원인:** {cause}")
            prob_df = pd.DataFrame({'패턴': le.classes_, '확률': proba}).sort_values('확률', ascending=True)
            figb, axb = plt.subplots(figsize=(4.5, 3.0))
            colors = ['#27D3EE' if pat == pred_cls else '#C7D3DF' for pat in prob_df['패턴']]
            axb.barh(prob_df['패턴'], prob_df['확률'], color=colors)
            axb.set_xlim(0, 1)
            for i, v in enumerate(prob_df['확률']):
                axb.text(min(v + 0.02, 0.92), i, f'{v:.1%}', va='center', fontsize=8)
            axb.set_xlabel('Probability'); axb.spines[['top', 'right']].set_visible(False)
            figb.tight_layout()
            st.pyplot(figb, use_container_width=True)
 
        # ── Grad-CAM 근거 시각화 ──
        st.markdown("---")
        st.markdown("####  Grad-CAM — 모델이 주목한 영역")
        try:
            hm = make_gradcam(arr, model, pred_index=int(proba.argmax()))
            ov = overlay_gradcam(arr, hm)
            g1, g2 = st.columns(2)
            with g1:
                fig2, ax2 = plt.subplots(figsize=(3.2,3.2)); ax2.imshow(arr, cmap='viridis'); ax2.axis('off'); ax2.set_title('원본')
                st.pyplot(fig2, use_container_width=False)
            with g2:
                fig3, ax3 = plt.subplots(figsize=(3.2,3.2)); ax3.imshow(ov); ax3.axis('off'); ax3.set_title('Grad-CAM 오버레이')
                st.pyplot(fig3, use_container_width=False)
            st.caption("빨강(JET)이 진할수록 예측에 크게 기여한 영역 — 결함 부위와 일치할수록 신뢰할 수 있는 판단")
        except Exception as e:
            st.info("Grad-CAM은 last_conv 층이 있는 모델에서 동작합니다. (노트북 최신 모델 사용)")
 
# ============================================================
# 탭3 — 클래스 정보 (예시 그림 + 형태 설명 + 공정 원인)
# ============================================================
with tab3:
    st.subheader("결함 패턴별 예시 · 형태 · 추정 공정 원인")
    st.caption("색 규칙 — 보라: 웨이퍼 밖 / 청록: 정상 die / 노랑: 불량 die")
    for name, (appear, cause) in PATTERN_INFO.items():
        ic1, ic2 = st.columns([1, 4])
        with ic1:
            st.pyplot(thumb(name))
        with ic2:
            st.markdown(f"### {name}")
            st.markdown(f"**결함 현상** — {appear}")
            st.markdown(f"**추정 공정 원인 / 점검 포인트** — {cause}")
        st.divider()
 
# ============================================================
# 탭4 — 모델 정보
# ============================================================
with tab4:
    st.subheader("모델 정보")
    if model_ready:
        st.write("**클래스:**", list(le.classes_))
        st.write("**입력 형태:** (64, 64, 1) 웨이퍼맵")
        st.write("**구조:** Conv2D(16) → Conv2D(32) → Conv2D(64) → Dropout → Flatten → Dense(128) → Softmax")
        st.write("**손실/최적화:** SparseCategoricalCrossentropy / Adam, EarlyStopping(patience=3)")
    st.caption("웨이퍼맵(1채널)에 맞게 확장")