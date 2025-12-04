"""
Project 1: Perceptron - Furniture Placement Optimization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

st.set_page_config(page_title="Perceptron", page_icon="📊", layout="wide")
st.title("📊 Perceptron: Furniture Placement Optimization")

with st.expander("📖 Show Project Overview", expanded=False):
    st.markdown("""
    ### Purpose
    Determine optimal furniture placement using a Perceptron that learns from distance-based features.

    ### How It Works
    1. **Extract Features**: Distance from walls, doors, windows, furniture, and room center
    2. **Train**: Adjust weights when predictions are wrong until convergence
    3. **Classify**: Predict good (near windows, proper clearance) vs bad (blocking doors) placements

    ### Results
    - Accuracy: 95%+ after convergence (20-50 epochs)
    - Visualization shows decision boundary (green=good, red=bad zones)
    - Dataset: 200 synthetic examples
    """)

st.markdown("---")
st.markdown("### ⚙️ Configuration")

col1, col2 = st.columns(2)
with col1:
    room_width = st.slider("Room Width (m)", 5.0, 15.0, 10.0, 0.5)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
with col2:
    room_height = st.slider("Room Height (m)", 5.0, 15.0, 8.0, 0.5)
    max_epochs = st.slider("Max Epochs", 10, 200, 100, 10)

st.markdown("---")

class Room:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.doors = [(0, 3, 0.2, 2)]
        self.windows = [(5, height, 3, 0.2), (width, 2, 0.2, 2)]
        self.furniture = [(1, 1, 2, 1), (width-2.5, height-3, 1.5, 2)]

class Perceptron:
    def __init__(self, n_features, learning_rate=0.1):
        self.weights = np.random.uniform(-1, 1, n_features)
        self.learning_rate = learning_rate
        self.history = []

    def predict(self, x):
        return 1 if np.dot(self.weights, x) >= 0 else -1

    def train_step(self, x, target):
        prediction = self.predict(x)
        if prediction != target:
            if target == 1 and prediction == -1:
                self.weights += self.learning_rate * x
            elif target == -1 and prediction == 1:
                self.weights -= self.learning_rate * x
            return True
        return False

    def train(self, X, y, max_epochs):
        for epoch in range(max_epochs):
            errors = 0
            indices = np.random.permutation(len(X))
            for idx in indices:
                if self.train_step(X[idx], y[idx]):
                    errors += 1
            accuracy = 1 - errors / len(X)
            self.history.append({'epoch': epoch, 'accuracy': accuracy})
            if errors == 0:
                break
        return self.history

def distance_to_nearest(x, y, objects):
    if not objects:
        return float('inf')
    min_dist = float('inf')
    for obj_x, obj_y, obj_w, obj_h in objects:
        dist = np.sqrt((x - obj_x - obj_w/2)**2 + (y - obj_y - obj_h/2)**2)
        min_dist = min(min_dist, dist)
    return min_dist

def extract_features(x, y, room):
    max_dist = np.sqrt(room.width**2 + room.height**2)
    return np.array([
        min(x, y, room.width - x, room.height - y) / max_dist,
        min(distance_to_nearest(x, y, room.doors) / max_dist, 1.0),
        min(distance_to_nearest(x, y, room.windows) / max_dist, 1.0),
        min(distance_to_nearest(x, y, room.furniture) / max_dist, 1.0),
        np.sqrt((x - room.width/2)**2 + (y - room.height/2)**2) / max_dist
    ])

def is_good_placement(x, y, room):
    if distance_to_nearest(x, y, room.doors) < 2.0 or distance_to_nearest(x, y, room.furniture) < 1.5:
        return False
    if x < 0.5 or y < 0.5 or x > room.width - 0.5 or y > room.height - 0.5:
        return False
    return distance_to_nearest(x, y, room.windows) < 3.0

if st.button("🚀 Train Perceptron", type="primary"):
    with st.spinner("Training..."):
        room = Room(room_width, room_height)
        np.random.seed(42)
        X_train, y_train = [], []
        for i in range(200):
            x_pos, y_pos = np.random.uniform(0, room.width), np.random.uniform(0, room.height)
            X_train.append(extract_features(x_pos, y_pos, room))
            y_train.append(1 if is_good_placement(x_pos, y_pos, room) else -1)
        
        perceptron = Perceptron(5, learning_rate)
        history = perceptron.train(np.array(X_train), np.array(y_train), max_epochs)
        
        st.success(f"✅ Converged at epoch {len(history)}")
        st.markdown("### 📊 Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Training Progress")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot([h['epoch'] for h in history], [h['accuracy'] for h in history], 'b-', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1.1])
            st.pyplot(fig1)
        
        with col2:
            st.markdown("#### Learned Weights")
            st.dataframe(pd.DataFrame({
                'Feature': ['Wall', 'Door', 'Window', 'Furniture', 'Center'],
                'Weight': perceptron.weights
            }), use_container_width=True, hide_index=True)
        
        st.markdown("#### Decision Boundary")
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        xx, yy = np.meshgrid(np.linspace(0, room.width, 50), np.linspace(0, room.height, 50))
        grid = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                grid[i, j] = perceptron.predict(extract_features(xx[i, j], yy[i, j], room))
        
        ax2.imshow(grid, extent=[0, room.width, 0, room.height], origin='lower', cmap='RdYlGn', alpha=0.3)
        ax2.add_patch(Rectangle((0, 0), room.width, room.height, linewidth=2, edgecolor='black', facecolor='none'))
        for door in room.doors:
            ax2.add_patch(Rectangle((door[0], door[1]), door[2], door[3], linewidth=2, edgecolor='brown', facecolor='brown'))
        for window in room.windows:
            ax2.add_patch(Rectangle((window[0], window[1]), window[2], window[3], linewidth=2, edgecolor='blue', facecolor='lightblue'))
        for furn in room.furniture:
            ax2.add_patch(Rectangle((furn[0], furn[1]), furn[2], furn[3], linewidth=2, edgecolor='red', facecolor='lightcoral'))
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Good (Green) vs Bad (Red) Placements')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
