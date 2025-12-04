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
    1. **CNN Architecture**: Improved deep architecture with batch normalization
       - 3 Conv blocks: Conv2D(32×2) → Conv2D(64×2) → Conv2D(128×2)
       - MaxPooling and Dropout after each block
       - Dense(128) → Output(10 classes)
    2. **Training**: Data augmentation (rotation, shifts, flips) + 15 epochs default
    3. **Inference**: Upload image or use CIFAR-10 samples for classification

    ### Results
    - Test Accuracy: 80%+ achievable with 3000+ samples/class
    - Dataset: CIFAR-10 (60,000 images, 32×32 RGB)
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
    st.markdown("### 📸 Quick Training & Classification")

    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = None
        st.session_state.cnn_trained = False

    col1, col2 = st.columns([1, 1])
    with col1:
        train_samples = st.slider("Training Samples (per class)", 500, 5000, 3000, 500)
    with col2:
        train_epochs = st.slider("Training Epochs", 5, 25, 15)

    if st.button("🚀 Train CNN Model", type="primary"):
        with st.spinner(f"Training CNN on {train_samples*10} CIFAR-10 images..."):
            # Load CIFAR-10 data
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()

            # Use subset for quick training
            indices = []
            for class_id in range(10):
                class_indices = np.where(y_train == class_id)[0][:train_samples]
                indices.extend(class_indices)
            indices = np.array(indices)

            X_subset = X_train[indices].astype('float32') / 255.0
            y_subset = keras.utils.to_categorical(y_train[indices], 10)

            # Data augmentation for better accuracy
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
            datagen.fit(X_subset)

            # Improved CNN model architecture
            model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.2),

                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.3),

                keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.4),

                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])

            # Use Adam optimizer with learning rate decay
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # Train with data augmentation
            history = model.fit(datagen.flow(X_subset, y_subset, batch_size=64),
                              epochs=train_epochs,
                              validation_split=0.15,
                              verbose=0,
                              steps_per_epoch=len(X_subset) // 64)

            # Evaluate on test set (larger subset for accurate measurement)
            X_test_subset = X_test[:2000].astype('float32') / 255.0
            y_test_subset = keras.utils.to_categorical(y_test[:2000], 10)
            test_loss, test_acc = model.evaluate(X_test_subset, y_test_subset, verbose=0)

            # Also get training accuracy
            train_loss, train_acc = model.evaluate(X_subset, y_subset, verbose=0)

            st.session_state.cnn_model = model
            st.session_state.cnn_trained = True

            st.success(f"✅ Training complete! Test Accuracy: {test_acc*100:.2f}% | Training Accuracy: {train_acc*100:.2f}%")
            st.info(f"📊 Trained on {len(X_subset)} images for {train_epochs} epochs")

            # Show accuracy metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
            with col2:
                st.metric("Test Accuracy", f"{test_acc*100:.2f}%")
            with col3:
                st.metric("Model Size", f"{len(X_subset):,} samples")

    if st.session_state.cnn_model and st.session_state.cnn_trained:
        st.markdown("### 🖼️ Classify Image")
        uploaded = st.file_uploader("Upload image (will be resized to 32×32)", type=['png', 'jpg', 'jpeg'])
        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Image**")
                st.image(image, width=200)

            # Resize and predict
            image_resized = image.resize((32, 32))
            img_array = np.array(image_resized).astype('float32') / 255.0

            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]

            img_array = np.expand_dims(img_array, axis=0)

            predictions = st.session_state.cnn_model.predict(img_array, verbose=0)[0]
            class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                          'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100

            with col2:
                st.markdown("**Prediction**")
                st.success(f"### {class_names[predicted_class]}")
                st.metric("Confidence", f"{confidence:.1f}%")

                # Show top 3 predictions
                st.markdown("**Top 3 Predictions:**")
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                for idx in top_3_idx:
                    st.write(f"{class_names[idx]}: {predictions[idx]*100:.1f}%")

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
