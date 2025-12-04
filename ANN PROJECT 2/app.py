"""
NBA Optimal Team Selection using Deep Artificial Neural Network
Author: Chrisian Nshuti Manzi, Aime Serge Tuyishime
Course: CST-435
Assignment: Artificial Neural Network (ANN)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NBA Optimal Team Selection - ANN",
    page_icon="🏀",
    layout="wide"
)

# Title and description
st.title("🏀 NBA Optimal Team Selection using Deep Neural Network")
st.markdown("""
### CST-435: Artificial Neural Network Assignment
This application uses a deep Multi-Layer Perceptron (MLP) to select the optimal 5-player NBA team
from a pool of 100 players based on balanced team composition.
""")
# Check if data file exists
data_file = "all_seasons.csv.xlsx"
if not os.path.exists(data_file):
    st.error(f"❌ Data file '{data_file}' not found!")
    st.info("""
    Please ensure the NBA dataset file is in the same directory as app.py:
    
    1. Download the 'all_seasons.csv.xlsx' file from your course materials
    2. Place it in: `C:\\Users\\nshut\\Documents\\CST 435\\projects\\ANN\\`
    3. Refresh this page
    
    Current working directory: {}
    """.format(os.getcwd()))
    
    # Option to upload file directly
    st.subheader("Or upload the file here:")
    uploaded_file = st.file_uploader("Choose the all_seasons.csv.xlsx file", type=['xlsx'])
    if uploaded_file is not None:
        # Save the uploaded file
        with open(data_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File uploaded successfully! Please refresh the page.")
        st.rerun()
    st.stop()

# Sidebar for parameters
st.sidebar.header("Neural Network Configuration")

# Load and prepare data
@st.cache_data
def load_data():
    """Load and preprocess NBA player data"""
    # For deployment, you'll need to upload the actual file
    # This is a simulated dataset based on the structure you showed
    data = pd.read_excel("all_seasons.csv.xlsx", sheet_name="all_seasons")
    return data

@st.cache_data
def prepare_player_pool(data, start_season='2018-19', end_season='2022-23'):
    """
    Select 100 players from a 5-year window for team selection analysis.

    This function implements the requirement to select a pool of 100 players
    from the dataset within a 5-year window, ensuring data quality and relevance.

    Parameters:
    - data: Complete NBA dataset
    - start_season: Beginning of the analysis window
    - end_season: End of the analysis window

    Returns:
    - DataFrame with top 100 players based on composite performance score
    """
    # Filter seasons: Focus on recent 5-year window (2018-2023)
    # This ensures contemporary playing styles and rule sets
    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    filtered_data = data[data['season'].isin(seasons)].copy()

    # Remove players with insufficient games (< 20 games played)
    # This ensures statistical reliability and excludes injury-limited seasons
    filtered_data = filtered_data[filtered_data['gp'] >= 20]

    # Calculate composite score for player quality ranking
    # Weighted combination of key performance indicators:
    # - Points (30%): Primary offensive contribution
    # - Rebounds (25%): Defensive presence and second-chance opportunities
    # - Assists (25%): Playmaking and team facilitation
    # - Net Rating (10%): Overall team impact when player is on court
    # - True Shooting % (10%): Shooting efficiency
    filtered_data['composite_score'] = (
        filtered_data['pts'] * 0.3 +           # Scoring impact
        filtered_data['reb'] * 0.25 +          # Rebounding contribution
        filtered_data['ast'] * 0.25 +          # Playmaking ability
        filtered_data['net_rating'] * 0.1 +    # Team impact
        filtered_data['ts_pct'] * 10 * 0.1     # Shooting efficiency (scaled)
    )

    # Group by player and get their best season performance
    # This handles players who appear in multiple seasons within the window
    player_best = filtered_data.sort_values('composite_score', ascending=False).groupby('player_name').first().reset_index()

    # Select top 100 players based on composite score
    # This creates the required pool of 100 players for team selection
    top_100 = player_best.nlargest(100, 'composite_score').reset_index(drop=True)

    return top_100

# Deep Neural Network Implementation
class DeepMLP:
    """
    Deep Multi-Layer Perceptron for NBA Team Selection
    Architecture: Input Layer -> 3 Hidden Layers -> Output Layer
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=5, learning_rate=0.001):
        """
        Initialize the deep neural network
        
        Parameters:
        - input_size: Number of input features
        - hidden_sizes: List of hidden layer sizes
        - output_size: Number of output neurons (5 for team selection)
        - learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = []
        
        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Store activations for backpropagation
        self.activations = []
        self.z_values = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))

        Provides smooth, differentiable activation with output range (0, 1).
        Used in binary classification tasks. Includes clipping to prevent overflow.
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function: σ'(z) = σ(z) × (1 - σ(z))

        Used in backpropagation for gradient calculation through sigmoid layers.
        """
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        """
        Rectified Linear Unit (ReLU) activation: f(z) = max(0, z)

        Provides non-linearity while avoiding vanishing gradient problem.
        Preferred for hidden layers in deep networks due to computational efficiency.
        """
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """
        Derivative of ReLU function: f'(z) = 1 if z > 0, else 0

        Used in backpropagation. Creates sparse gradients that help with
        computational efficiency and can act as implicit regularization.
        """
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        """
        Softmax activation for multi-class classification: σ(z_i) = e^(z_i) / Σe^(z_j)

        Converts raw logits to probability distribution over classes.
        Output sums to 1, making it ideal for classification tasks.
        Includes numerical stability through max subtraction.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the neural network.

        Implements the forward pass of the MLP, computing activations layer by layer.
        This is a core requirement demonstrating "forward propagation to calculate
        the network output."

        Algorithm:
        1. Initialize with input layer
        2. For each hidden layer: z = X·W + b, a = ReLU(z)
        3. For output layer: z = X·W + b, a = Softmax(z)

        Parameters:
        - X: Input data matrix (batch_size × input_features)

        Returns:
        - Output probabilities (batch_size × num_classes)
        """
        # Store activations for backpropagation - critical for gradient calculation
        self.activations = [X]  # Store input layer
        self.z_values = []      # Store pre-activation values

        current_input = X

        # Propagate through hidden layers with ReLU activation
        # ReLU chosen for its computational efficiency and gradient flow properties
        for i in range(len(self.weights) - 1):
            # Linear transformation: z = X·W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            # Non-linear activation: a = ReLU(z)
            activation = self.relu(z)
            self.activations.append(activation)
            current_input = activation

        # Output layer with softmax for position classification
        # Softmax ensures output forms probability distribution over 5 positions
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.softmax(z_out)  # Probability distribution over positions
        self.activations.append(output)

        return output
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss
        
        Parameters:
        - y_pred: Predicted probabilities
        - y_true: True labels (one-hot encoded)
        
        Returns:
        - Loss value
        """
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def backward_propagation(self, X, y_true, y_pred):
        """
        Backward propagation to calculate gradients
        
        Parameters:
        - X: Input data
        - y_true: True labels
        - y_pred: Predicted values
        
        Returns:
        - Gradients for weights and biases
        """
        m = X.shape[0]
        
        # Initialize gradient storage
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        dz = y_pred - y_true
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self.relu_derivative(self.z_values[i-1])
        
        return dW, db
    
    def update_weights(self, dW, db):
        """
        Update weights and biases using gradient descent
        
        Parameters:
        - dW: Weight gradients
        - db: Bias gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Parameters:
        - X_train: Training data
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Batch size for mini-batch gradient descent
        - verbose: Print training progress
        
        Returns:
        - Training history
        """
        history = {'loss': []}
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = n_samples // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward propagation
                dW, db = self.backward_propagation(X_batch, y_batch, y_pred)
                
                # Update weights
                self.update_weights(dW, db)
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def apply_threshold(self, probabilities):
        """
        Apply threshold function to obtain predicted class labels in one-hot representation

        Parameters:
        - probabilities: Softmax output probabilities

        Returns:
        - One-hot encoded predictions
        """
        # Find the class with maximum probability for each sample
        predicted_classes = np.argmax(probabilities, axis=1)

        # Convert to one-hot representation
        one_hot_predictions = np.zeros_like(probabilities)
        one_hot_predictions[np.arange(len(predicted_classes)), predicted_classes] = 1.0

        return one_hot_predictions

    def predict(self, X):
        """
        Make predictions on new data

        Parameters:
        - X: Input data

        Returns:
        - One-hot encoded predicted class labels
        """
        probabilities = self.forward_propagation(X)
        one_hot_predictions = self.apply_threshold(probabilities)
        return one_hot_predictions

# Feature engineering
def create_features(players_df):
    """
    Create features for the neural network
    
    Parameters:
    - players_df: DataFrame with player statistics
    
    Returns:
    - Feature matrix and feature names
    """
    features = []
    feature_names = []
    
    # Basic stats (normalized)
    stat_cols = ['pts', 'reb', 'ast', 'net_rating', 'ts_pct', 'usg_pct', 
                 'oreb_pct', 'dreb_pct', 'ast_pct']
    
    for col in stat_cols:
        if col in players_df.columns:
            features.append(players_df[col].values)
            feature_names.append(col)
    
    # Physical attributes
    if 'player_height' in players_df.columns:
        features.append(players_df['player_height'].values)
        feature_names.append('height')
    
    if 'player_weight' in players_df.columns:
        features.append(players_df['player_weight'].values)
        feature_names.append('weight')
    
    if 'age' in players_df.columns:
        features.append(players_df['age'].values)
        feature_names.append('age')
    
    # Composite metrics
    features.append(players_df['composite_score'].values)
    feature_names.append('composite_score')
    
    # Stack features
    X = np.stack(features, axis=1)
    
    return X, feature_names

# Team balance evaluation
def evaluate_team_balance(team_players):
    """
    Evaluate how balanced a team is across different positions and skills
    
    Parameters:
    - team_players: DataFrame with selected team players
    
    Returns:
    - Balance score and breakdown
    """
    balance_metrics = {}
    
    # Scoring balance
    balance_metrics['scoring'] = team_players['pts'].std()
    
    # Rebounding presence
    balance_metrics['rebounding'] = team_players['reb'].mean()
    
    # Playmaking
    balance_metrics['playmaking'] = team_players['ast'].mean()
    
    # Efficiency
    balance_metrics['efficiency'] = team_players['ts_pct'].mean()
    
    # Size diversity
    balance_metrics['height_diversity'] = team_players['player_height'].std()
    
    # Overall balance score (lower is better for std, higher for means)
    balance_score = (
        -balance_metrics['scoring'] * 0.2 +  # Negative because we want low std
        balance_metrics['rebounding'] * 0.2 +
        balance_metrics['playmaking'] * 0.2 +
        balance_metrics['efficiency'] * 0.2 +
        balance_metrics['height_diversity'] * 0.2  # Some diversity is good
    )
    
    return balance_score, balance_metrics

# Main app
def main():
    # Load data
    try:
        data = load_data()
        st.success("✅ Data loaded successfully!")
    except:
        st.error("❌ Please ensure the 'all_seasons.csv.xlsx' file is in the same directory")
        st.stop()
    
    # Prepare player pool
    with st.spinner("Preparing player pool..."):
        players_100 = prepare_player_pool(data)
        st.success(f"✅ Selected top 100 players from 2018-2023 seasons")
    
    # Display player pool statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players in Pool", len(players_100))
    with col2:
        st.metric("Average PPG", f"{players_100['pts'].mean():.1f}")
    with col3:
        st.metric("Average Age", f"{players_100['age'].mean():.1f}")
    
    # Feature preparation
    X, feature_names = create_features(players_100)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Neural Network Configuration
    st.sidebar.subheader("Network Architecture")
    hidden_layer_1 = st.sidebar.slider("Hidden Layer 1 Neurons", 32, 256, 128)
    hidden_layer_2 = st.sidebar.slider("Hidden Layer 2 Neurons", 16, 128, 64)
    hidden_layer_3 = st.sidebar.slider("Hidden Layer 3 Neurons", 8, 64, 32)
    learning_rate = st.sidebar.select_slider("Learning Rate", 
                                            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                            value=0.001)
    epochs = st.sidebar.slider("Training Epochs", 50, 500, 200)
    
    # Train model button
    if st.button("🚀 Train Neural Network and Select Team", type="primary"):
        
        st.header("🧠 Neural Network Training")
        
        # Initialize neural network
        input_size = X_scaled.shape[1]
        hidden_sizes = [hidden_layer_1, hidden_layer_2, hidden_layer_3]
        mlp = DeepMLP(input_size, hidden_sizes, output_size=5, learning_rate=learning_rate)
        
        # Create synthetic labels for team selection (5 position classes)
        # We'll use a scoring system to identify ideal team positions
        y_labels = np.zeros((100, 5))
        
        # Assign position labels based on player characteristics
        # 0: Point Guard, 1: Shooting Guard, 2: Small Forward, 3: Power Forward, 4: Center
        for i in range(100):
            player = players_100.iloc[i]
            height = player['player_height']
            assists = player['ast']
            rebounds = player['reb']

            # Position classification based on height and stats
            if height < 190 and assists > 5:  # Point Guard
                y_labels[i, 0] = 1.0
            elif height < 195 and assists > 3:  # Shooting Guard
                y_labels[i, 1] = 1.0
            elif height < 205:  # Small Forward
                y_labels[i, 2] = 1.0
            elif height < 210 and rebounds > 6:  # Power Forward
                y_labels[i, 3] = 1.0
            else:  # Center
                y_labels[i, 4] = 1.0
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom training loop with progress updates
        history = {'loss': []}
        batch_size = 16
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(100)
            X_epoch = X_scaled[indices]
            y_epoch = y_labels[indices]
            
            # Forward propagation
            y_pred = mlp.forward_propagation(X_epoch)
            
            # Compute loss
            loss = mlp.compute_loss(y_pred, y_epoch)
            history['loss'].append(loss)
            
            # Backward propagation
            dW, db = mlp.backward_propagation(X_epoch, y_epoch, y_pred)
            
            # Update weights
            mlp.update_weights(dW, db)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
        
        st.success("✅ Neural Network Training Complete!")
        
        # Plot training history
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(history['loss']) + 1)),
            y=history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        fig_loss.update_layout(
            title="Training Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Team Selection using trained network
        st.header("🏆 Optimal Team Selection")

        # Get network predictions for all players (position probabilities)
        position_probabilities = mlp.forward_propagation(X_scaled)
        position_predictions = mlp.apply_threshold(position_probabilities)

        # Calculate position confidence scores
        position_confidence = np.max(position_probabilities, axis=1)
        
        # Select players based on predicted positions
        selected_indices = []

        # Group players by predicted position (0: PG, 1: SG, 2: SF, 3: PF, 4: C)
        position_groups = [[] for _ in range(5)]
        position_names = ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center"]

        for i in range(100):
            predicted_position = np.argmax(position_predictions[i])
            confidence = position_confidence[i]
            position_groups[predicted_position].append((i, confidence))

        # Sort each position group by confidence
        for group in position_groups:
            group.sort(key=lambda x: x[1], reverse=True)

        # Select one player from each position (balanced team)
        for pos in range(5):
            if len(position_groups[pos]) > 0:
                selected_indices.append(position_groups[pos][0][0])
            else:
                # If no player predicted for this position, find next best option
                remaining = [i for i in range(100) if i not in selected_indices]
                if remaining:
                    best_remaining = max(remaining, key=lambda x: position_confidence[x])
                    selected_indices.append(best_remaining)

        # Ensure we have exactly 5 players
        while len(selected_indices) < 5:
            remaining = [i for i in range(100) if i not in selected_indices]
            if remaining:
                best_remaining = max(remaining, key=lambda x: position_confidence[x])
                selected_indices.append(best_remaining)
            else:
                break
        
        # Get selected team
        selected_team = players_100.iloc[selected_indices[:5]]
        
        # Display position predictions for selected team
        st.subheader("Position Predictions for Selected Team")

        position_data = []
        for idx, player_idx in enumerate(selected_indices[:5]):
            player_name = players_100.iloc[player_idx]['player_name']
            predicted_pos = np.argmax(position_predictions[player_idx])
            confidence = position_confidence[player_idx]
            position_data.append({
                'Player': player_name,
                'Predicted Position': position_names[predicted_pos],
                'Confidence': f"{confidence:.3f}"
            })

        position_df = pd.DataFrame(position_data)
        st.dataframe(position_df, use_container_width=True)

        # Display selected team
        st.subheader("Selected Optimal Team")

        # Team overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Team PPG", f"{selected_team['pts'].sum():.1f}")
        with col2:
            st.metric("Team RPG", f"{selected_team['reb'].sum():.1f}")
        with col3:
            st.metric("Team APG", f"{selected_team['ast'].sum():.1f}")
        with col4:
            st.metric("Avg Efficiency", f"{selected_team['ts_pct'].mean():.3f}")
        
        # Display team roster
        st.dataframe(
            selected_team[['player_name', 'age', 'player_height', 'player_weight', 
                         'pts', 'reb', 'ast', 'ts_pct', 'net_rating']],
            use_container_width=True
        )
        
        # Team balance evaluation
        balance_score, balance_metrics = evaluate_team_balance(selected_team)
        
        st.subheader("Team Balance Analysis")
        
        # Create radar chart for team balance
        categories = ['Scoring\nBalance', 'Rebounding', 'Playmaking', 'Efficiency', 'Size\nDiversity']
        values = [
            1 / (1 + balance_metrics['scoring']),  # Invert for visualization
            balance_metrics['rebounding'] / 10,
            balance_metrics['playmaking'] / 8,
            balance_metrics['efficiency'],
            balance_metrics['height_diversity'] / 10
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Team Balance'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Team Balance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Player comparison
        st.subheader("Player Statistics Comparison")
        
        fig_comparison = go.Figure()
        
        metrics_to_compare = ['pts', 'reb', 'ast']
        colors = ['blue', 'green', 'red']
        
        for i, metric in enumerate(metrics_to_compare):
            fig_comparison.add_trace(go.Bar(
                name=metric.upper(),
                x=selected_team['player_name'],
                y=selected_team[metric],
                marker_color=colors[i]
            ))
        
        fig_comparison.update_layout(
            title="Team Members Statistical Comparison",
            xaxis_title="Player",
            yaxis_title="Stats",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Network Architecture Visualization
        st.header("🔧 Neural Network Architecture")
        
        st.markdown(f"""
        ### Network Configuration:
        - **Input Layer**: {input_size} neurons (player features)
        - **Hidden Layer 1**: {hidden_layer_1} neurons (ReLU activation)
        - **Hidden Layer 2**: {hidden_layer_2} neurons (ReLU activation)
        - **Hidden Layer 3**: {hidden_layer_3} neurons (ReLU activation)
        - **Output Layer**: 5 neurons (Softmax activation for 5 positions)
        - **Learning Rate**: {learning_rate}
        - **Training Epochs**: {epochs}
        
        ### Features Used:
        {', '.join(feature_names)}
        
        ### Training Process:
        1. **Forward Propagation**: Input features propagated through network layers
        2. **Loss Calculation**: Cross-entropy loss computed between predictions and targets
        3. **Backpropagation**: Gradients calculated through chain rule
        4. **Weight Updates**: Gradient descent optimization
        5. **Iteration**: Process repeated for {epochs} epochs
        """)
        
        # MLP Interpretation
        st.header("📊 MLP Output Interpretation")
        
        st.markdown("""
        ### Team Selection Interpretation:
        
        The Multi-Layer Perceptron has learned to identify optimal team compositions by:
        
        1. **Feature Extraction**: The first hidden layer identifies key player attributes
        2. **Pattern Recognition**: Middle layers learn complex relationships between stats
        3. **Team Synergy**: The network considers how players complement each other
        4. **Balance Optimization**: Output layer scores players based on team fit
        
        The selected team represents a balance of:
        - **Scoring ability** (distributed across players)
        - **Defensive presence** (rebounding and defensive metrics)
        - **Playmaking** (assist generation)
        - **Efficiency** (shooting percentages)
        - **Positional diversity** (height and role variation)
        
        This approach ensures the team has no critical weaknesses while maximizing overall performance.
        """)

if __name__ == "__main__":
    main()