"""
Project 2: Deep ANN - NBA Team Selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Deep ANN - NBA", page_icon="🏀", layout="wide")
st.title("🏀 Deep ANN: NBA Team Selection")

with st.expander("📖 Show Project Overview", expanded=False):
    st.markdown("""
    ### Purpose
    Select optimal 5-player NBA team from 100 players using a Deep Multi-Layer Perceptron that balances positions and skills.

    ### How It Works
    1. **Data Preparation**: Select top 100 NBA players from 2018-2023 seasons
    2. **Feature Extraction**: Player stats (points, rebounds, assists, efficiency, physical attributes)
    3. **Neural Network**: 3 hidden layers (128→64→32 neurons) with ReLU activation
    4. **Position Classification**: Classify players into 5 positions (PG, SG, SF, PF, C)
    5. **Team Selection**: Pick one player from each position for balanced team

    ### Results
    - Architecture: 13 input features → 128 → 64 → 32 → 5 output positions
    - Training: Adam optimizer with cross-entropy loss
    - Selection: Balanced team across all positions
    - Dataset: 100 top NBA players (2018-2023)
    """)

st.markdown("---")

# Check data file
data_file = "all_seasons.csv.xlsx"
if not os.path.exists(data_file):
    st.error(f"❌ Data file '{data_file}' not found!")
    st.info("Please place 'all_seasons.csv.xlsx' in the project directory.")
    uploaded_file = st.file_uploader("Upload the file here:", type=['xlsx'])
    if uploaded_file:
        with open(data_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("DONE: File uploaded! Refresh the page.")
        st.rerun()
    st.stop()

@st.cache_data
def load_data():
    return pd.read_excel("all_seasons.csv.xlsx", sheet_name="all_seasons")

@st.cache_data
def prepare_player_pool(data):
    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    filtered = data[data['season'].isin(seasons)].copy()
    filtered = filtered[filtered['gp'] >= 20]
    filtered['composite_score'] = (filtered['pts'] * 0.3 + filtered['reb'] * 0.25 +
                                   filtered['ast'] * 0.25 + filtered['net_rating'] * 0.1 +
                                   filtered['ts_pct'] * 10 * 0.1)
    player_best = filtered.sort_values('composite_score', ascending=False).groupby('player_name').first().reset_index()
    return player_best.nlargest(100, 'composite_score').reset_index(drop=True)

class DeepMLP:
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=5, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.weights, self.biases = [], []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        self.activations, self.z_values = [], []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        current = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)
            current = activation
        z_out = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.softmax(z_out)
        self.activations.append(output)
        return output

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0]
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        dz = y_pred - y_true
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self.relu_derivative(self.z_values[i-1])
        return dW, db

    def update_weights(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def predict(self, X):
        probabilities = self.forward_propagation(X)
        predicted_classes = np.argmax(probabilities, axis=1)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(predicted_classes)), predicted_classes] = 1.0
        return one_hot, probabilities

def create_features(df):
    features, names = [], []
    stat_cols = ['pts', 'reb', 'ast', 'net_rating', 'ts_pct', 'usg_pct', 'oreb_pct', 'dreb_pct', 'ast_pct']
    for col in stat_cols:
        if col in df.columns:
            features.append(df[col].values)
            names.append(col)
    if 'player_height' in df.columns:
        features.append(df['player_height'].values)
        names.append('height')
    if 'player_weight' in df.columns:
        features.append(df['player_weight'].values)
        names.append('weight')
    if 'age' in df.columns:
        features.append(df['age'].values)
        names.append('age')
    features.append(df['composite_score'].values)
    names.append('composite_score')
    return np.stack(features, axis=1), names

try:
    data = load_data()
    st.success("DONE: Data loaded successfully!")
except:
    st.error("❌ Error loading data")
    st.stop()

with st.spinner("Preparing player pool..."):
    players_100 = prepare_player_pool(data)
    st.success(f"DONE: Selected top 100 players")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Players in Pool", len(players_100))
with col2:
    st.metric("Average PPG", f"{players_100['pts'].mean():.1f}")
with col3:
    st.metric("Average Age", f"{players_100['age'].mean():.1f}")

X, feature_names = create_features(players_100)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.markdown("### ⚙️ Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    h1 = st.slider("Hidden Layer 1", 32, 256, 128)
    h2 = st.slider("Hidden Layer 2", 16, 128, 64)
with col2:
    h3 = st.slider("Hidden Layer 3", 8, 64, 32)
    lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
with col3:
    epochs = st.slider("Epochs", 50, 500, 200)

st.markdown("---")

if st.button("🚀 Train & Select Team", type="primary"):
    st.markdown("### 🧠 Training")
    mlp = DeepMLP(X_scaled.shape[1], [h1, h2, h3], 5, lr)
    y_labels = np.zeros((100, 5))
    for i in range(100):
        player = players_100.iloc[i]
        height, assists, rebounds = player['player_height'], player['ast'], player['reb']
        if height < 190 and assists > 5:
            y_labels[i, 0] = 1.0
        elif height < 195 and assists > 3:
            y_labels[i, 1] = 1.0
        elif height < 205:
            y_labels[i, 2] = 1.0
        elif height < 210 and rebounds > 6:
            y_labels[i, 3] = 1.0
        else:
            y_labels[i, 4] = 1.0

    progress_bar = st.progress(0)
    status = st.empty()
    history = {'loss': []}
    for epoch in range(epochs):
        indices = np.random.permutation(100)
        y_pred = mlp.forward_propagation(X_scaled[indices])
        loss = mlp.compute_loss(y_pred, y_labels[indices])
        history['loss'].append(loss)
        dW, db = mlp.backward_propagation(X_scaled[indices], y_labels[indices], y_pred)
        mlp.update_weights(dW, db)
        progress_bar.progress((epoch + 1) / epochs)
        status.text(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    st.success("DONE: Training complete!")

    st.markdown("### 📊 Results")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(history['loss']) + 1)), y=history['loss'],
                            mode='lines', name='Loss', line=dict(color='blue', width=2)))
    fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Selected Team")
    _, position_probs = mlp.predict(X_scaled)
    position_names = ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center"]
    position_groups = [[] for _ in range(5)]
    for i in range(100):
        pred_pos = np.argmax(position_probs[i])
        conf = np.max(position_probs[i])
        position_groups[pred_pos].append((i, conf))
    for group in position_groups:
        group.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for pos in range(5):
        if position_groups[pos]:
            selected.append(position_groups[pos][0][0])
        else:
            remaining = [i for i in range(100) if i not in selected]
            if remaining:
                selected.append(max(remaining, key=lambda x: np.max(position_probs[x])))

    team = players_100.iloc[selected[:5]]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Team PPG", f"{team['pts'].sum():.1f}")
    with col2:
        st.metric("Team RPG", f"{team['reb'].sum():.1f}")
    with col3:
        st.metric("Team APG", f"{team['ast'].sum():.1f}")
    with col4:
        st.metric("Avg Efficiency", f"{team['ts_pct'].mean():.3f}")

    st.dataframe(team[['player_name', 'age', 'player_height', 'player_weight', 'pts', 'reb', 'ast', 'ts_pct']],
                use_container_width=True)
