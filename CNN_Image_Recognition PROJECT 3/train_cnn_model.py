#!/usr/bin/env python3
"""
CNN Training Script for CIFAR-10 Image Classification
This script trains a CNN model and saves it for use in the Streamlit app.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("="*60)
    print("CNN Training for CIFAR-10 Image Classification")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("\n1. Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")

    print("\n2. Preprocessing data...")
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert labels to one-hot encoded vectors
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)

    print("   Data normalization complete")
    print(f"   Training data range: [{X_train.min()}, {X_train.max()}]")

    print("\n3. Building CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("   Model architecture:")
    model.summary()

    print("\n4. Training model...")
    print("   This will take 15-30 minutes depending on your hardware...")

    history = model.fit(
        X_train, y_train_categorical,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    print("\n5. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

    print(f"\n   Final Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print("\n6. Saving model...")
    # Save in HDF5 format
    model.save('cifar10_cnn_model.h5')
    print("   ✅ Model saved as 'cifar10_cnn_model.h5'")

    # Save in TensorFlow SavedModel format
    model.save('cifar10_cnn_model', save_format='tf')
    print("   ✅ Model saved as 'cifar10_cnn_model/' (SavedModel format)")

    print("\n7. Creating training history plot...")
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("   ✅ Training history saved as 'training_history.png'")

    print("\n" + "="*60)
    print("🎉 CNN Training Complete!")
    print("="*60)
    print("\nYour trained model is ready to use in the Streamlit app!")
    print("Now you can upload the model file to your deployed app.")
    print("\nFiles created:")
    print("- cifar10_cnn_model.h5 (main model file)")
    print("- cifar10_cnn_model/ (SavedModel directory)")
    print("- training_history.png (training visualization)")

if __name__ == "__main__":
    main()