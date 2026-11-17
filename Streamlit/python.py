# -*- coding: utf-8 -*-
"""Object Counting Web App (HW05 -> Streamlit)

파이프라인: grayscale -> median filter -> thresholding -> connected component labeling -> 통계/시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Counting", page_icon="🔬", layout="wide")


def otsu_threshold(img_u8: np.ndarray) -> int:
    """Otsu 방법으로 임계값 계산 (uint8 이미지)."""
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    w_b, sum_b, best, th = 0.0, 0.0, -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, th = var, t
    return th


@st.cache_data(show_spinner=False)
def run_pipeline(f, ksize, th_mode, th_manual, invert, min_area, connectivity):
    """전처리 -> 임계값 -> labeling -> 크기 필터링. 결과를 dict로 반환."""
    f_denoised = signal.medfilt2d(f, ksize) if ksize > 1 else f.copy()

    th_val = otsu_threshold(f_denoised) if th_mode == "Otsu (자동)" else th_manual
    mask = f_denoised > th_val
    if invert:
        mask = ~mask

    structure = np.ones((3, 3)) if connectivity == 8 else None  # None -> 4-connectivity
    labels, nums = nd.label(mask, structure=structure)

    if nums > 0:
        areas = np.bincount(labels.ravel(), minlength=nums + 1)[1:]
    else:
        areas = np.array([], dtype=int)

    # min_area 미만 object 제거 후 라벨 재정렬
    keep_ids = np.where(areas >= min_area)[0] + 1
    remap = np.zeros(nums + 1, dtype=np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1)
    labels = remap[labels]
    areas = areas[keep_ids - 1] if len(keep_ids) else np.array([], dtype=int)

    return dict(
        f_denoised=f_denoised,
        th_val=int(th_val),
        mask=mask,
        labels=labels,
        areas=areas,
        n_raw=int(nums),
        n=int(len(areas)),
    )


def fig_single(img, title, cmap="gray"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    return fig


# ----------------------------------------------------------------- UI
st.title("🔬 Object Counting Web App")
st.caption("이미지를 업로드하면 median filter → thresholding → connected component 분석으로 object 수를 계산합니다.")

with st.sidebar:
    st.header("파라미터")
    ksize = st.slider("Median filter 크기 (홀수)", 1, 21, 7, step=2)
    th_mode = st.radio("임계값 방식", ["수동", "Otsu (자동)"], index=0)
    th_manual = st.slider("임계값 (수동)", 0, 255, 128, disabled=(th_mode != "수동"))
    invert = st.checkbox("마스크 반전 (밝은 배경 위의 어두운 object)", value=False)
    connectivity = st.radio("Connectivity", [4, 8], index=1, horizontal=True)
    min_area = st.number_input("최소 object 크기 (pixel)", min_value=0, value=0, step=10,
                               help="이 값보다 작은 component는 noise로 간주하여 제외합니다.")

uploaded = st.file_uploader("이미지 업로드 (png, jpg, tif, bmp)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded is None:
    st.info("좌측 상단에서 이미지를 업로드해 주세요. (예: rice.tif)")
    st.stop()

try:
    img = Image.open(uploaded).convert("L")
except Exception as e:  # noqa: BLE001
    st.error(f"이미지를 열 수 없습니다: {e}")
    st.stop()

f = np.array(img)
res = run_pipeline(f, ksize, th_mode, th_manual, invert, int(min_area), connectivity)
areas, labels = res["areas"], res["labels"]

# --- 요약
st.subheader("결과 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Object 개수", res["n"])
c2.metric("사용된 임계값", res["th_val"])
if res["n"] > 0:
    c3.metric("최대 크기 (px)", int(areas.max()))
    c4.metric("최소 크기 (px)", int(areas.min()))
else:
    st.warning("검출된 object가 없습니다. 임계값 또는 반전 옵션을 조정해 보세요.")

# --- 단계별 영상
st.subheader("처리 단계")
tabs = st.tabs(["1. 원본", "2. 노이즈 제거", "3. 이진 마스크", "4. Labeling"])
with tabs[0]:
    st.pyplot(fig_single(f, "Input (grayscale)"))
with tabs[1]:
    st.pyplot(fig_single(res["f_denoised"], f"Median filter (k={ksize})"))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(res["f_denoised"].ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(res["th_val"], color="r", linestyle="--", label=f"threshold = {res['th_val']}")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
with tabs[2]:
    st.pyplot(fig_single(res["mask"], "Binary mask"))
with tabs[3]:
    shown = np.ma.masked_where(labels == 0, labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(f, cmap="gray")
    ax.imshow(shown, cmap="Spectral_r", alpha=0.7)
    ax.set_title(f"Labeled objects (n={res['n']})")
    ax.axis("off")
    st.pyplot(fig)

# --- 가장 큰/작은 component
if res["n"] > 0:
    st.subheader("가장 큰 / 가장 작은 component")
    i_max, i_min = int(np.argmax(areas)), int(np.argmin(areas))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(f, cmap="gray")
    axes[0].set_title("input image")
    axes[1].imshow(labels == i_max + 1, cmap="Spectral_r")
    axes[1].set_title(f"largest: index {i_max + 1}, {int(areas[i_max])} px")
    axes[2].imshow(labels == i_min + 1, cmap="Spectral_r")
    axes[2].set_title(f"smallest: index {i_min + 1}, {int(areas[i_min])} px")
    for a in axes:
        a.axis("off")
    st.pyplot(fig)

    # --- 크기 분포 및 표
    st.subheader("Object 크기 분포")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(areas, bins=min(30, max(5, res["n"] // 2)), color="steelblue", edgecolor="white")
        ax.axvline(areas.mean(), color="r", linestyle="--", label=f"mean = {areas.mean():.1f}")
        ax.set_xlabel("Area (pixels)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
    with col_b:
        st.write(f"평균 {areas.mean():.1f} px, 중앙값 {np.median(areas):.1f} px, 표준편차 {areas.std():.1f} px")
        table = {"index": np.arange(1, res["n"] + 1), "pixels": areas}
        st.dataframe(table, height=300, use_container_width=True)
        csv = "index,pixels\n" + "\n".join(f"{i},{a}" for i, a in zip(table["index"], areas))
        st.download_button("CSV 다운로드", csv, file_name="object_areas.csv", mime="text/csv")
