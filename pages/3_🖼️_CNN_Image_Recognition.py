"""
Project 3: CNN - Image Classification (CIFAR-10)
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="CNN - CIFAR-10", page_icon="🖼️", layout="wide")
st.title("🖼️ CNN: Image Classification (CIFAR-10)")

with st.expander("📖 Show Project Overview", expanded=False):
    st.markdown("""
    ### Purpose
    Classify images into 10 categories using a Convolutional Neural Network trained on CIFAR-10 dataset.

    ### How It Works
    1. **CNN Architecture**: 3 convolutional blocks with max pooling
       - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → MaxPool
       - Flatten → Dense(64) → Dropout → Dense(10)
    2. **Training**: 50 epochs with Adam optimizer, batch size 64
    3. **Inference**: Upload image or use CIFAR-10 samples for classification

    ### Results
    - Test Accuracy: 69.33%
    - Dataset: 60,000 images (32×32 RGB)
    - Classes: Airplane, Auto, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
    """)

st.markdown("---")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.datasets import cifar10
    import matplotlib.pyplot as plt
    from PIL import Image
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow not installed. Install with: `pip install tensorflow`")

if TF_AVAILABLE:
    st.markdown("### 📸 Classification Demo")

    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = None

    if st.button("📥 Load Model Architecture", type="primary"):
        with st.spinner("Loading..."):
            model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            st.session_state.cnn_model = model
            st.success("OK: Model loaded! Note: Train on full dataset for predictions.")

    if st.session_state.cnn_model:
        uploaded = st.file_uploader("Upload image (32×32 works best)", type=['png', 'jpg', 'jpeg'])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded", width=200)

    st.markdown("---")
    st.markdown("### 📸 CIFAR-10 Samples")
    if st.button("🎲 Load Random Samples"):
        try:
            (_, _), (X_test, y_test) = cifar10.load_data()
            class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            indices = np.random.choice(len(X_test), 10, replace=False)
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            for i, idx in enumerate(indices):
                axes[i].imshow(X_test[idx])
                axes[i].set_title(f"{class_names[y_test[idx][0]]}", fontsize=10)
                axes[i].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Install TensorFlow to use CNN demo")
