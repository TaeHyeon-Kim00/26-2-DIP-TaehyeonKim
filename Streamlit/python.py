# -*- coding: utf-8 -*-
"""Object Counting: 이미지를 업로드하면 object 개수를 세어 보여주는 Streamlit 앱 (HW05 기반)"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="centered")
st.title("Object Counting")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded is not None:
    f = np.array(Image.open(uploaded).convert("L"))

    f_denoised = signal.medfilt2d(f, 7)      # noise 제거
    mask = f_denoised > 128                  # thresholding
    labels, n = nd.label(mask)               # connected component

    st.metric("Object 개수", n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(f, cmap="gray")
    axes[0].set_title("input image")
    axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap="Spectral_r")
    axes[1].set_title(f"labeled objects (n={n})")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig)
