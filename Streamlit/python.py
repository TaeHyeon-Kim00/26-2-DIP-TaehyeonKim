import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

def iterative_threshold(img, delta_t=0.5):
    """반복적 global thresholding: T = (m1 + m2) / 2 를 수렴할 때까지 갱신."""
    t = img.mean()                      # 1. 초기 추정값
    while True:
        g1, g2 = img[img > t], img[img <= t]    # 2. T로 두 그룹 분리
        if g1.size == 0 or g2.size == 0:
            return t
        t_new = 0.5 * (g1.mean() + g2.mean())   # 3-4. 평균의 중간값으로 갱신
        if abs(t_new - t) < delta_t:            # 5. 수렴 확인
            return t_new
        t = t_new


st.set_page_config(page_title="Object Counting", layout="centered")
st.title("Object Counting")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded is not None:
    f = np.array(Image.open(uploaded).convert("L"))

    f_denoised = signal.medfilt2d(f, 7)      # noise 제거
    th = iterative_threshold(f_denoised)     # 반복적 임계값 추정
    mask = f_denoised > th                   # thresholding
    labels, n = nd.label(mask)               # connected component

    c1, c2 = st.columns(2)
    c1.metric("Object 개수", n)
    c2.metric("임계값 (iterative)", f"{th:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(f, cmap="gray")
    axes[0].set_title("input image")
    axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap="Spectral_r")
    axes[1].set_title(f"labeled objects (n={n})")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig)
