# NBA Optimal Team Selection using Deep Neural Network

**CST-435: Artificial Neural Network Assignment**
**Authors:** Christian Nshuti Manzi, Aime Serge Tuyishime

A deep Multi-Layer Perceptron (MLP) implementation for selecting optimal 5-player NBA teams from a pool of 100 players based on balanced team composition and position classification.

## 🚀 Live Demo

**Streamlit App**: [https://4rvzy3vjhzfyfhmrp2dsqx.streamlit.app/]

## 📋 Assignment Requirements Met

### ✅ Core ANN Components
- **Multi-Layer Perceptron (MLP)** - Complete implementation with input, 3 hidden layers, and output layer
- **Forward Propagation** - Layer-by-layer computation through the network
- **Backpropagation** - Gradient calculation and weight updates using chain rule
- **Error Calculation** - Cross-entropy loss function implementation
- **Threshold Function** - Argmax conversion to one-hot representation

### ✅ Technical Requirements
- **Deep ANN Architecture**: Input Layer → 3 Hidden Layers → Output Layer (5 classes)
- **Player Pool Selection**: Top 100 players from 5-year window (2018-2023)
- **Team Optimization**: Position-based balanced team selection
- **Output Interpretation**: Basketball position classification for optimal team building

### ✅ Implementation Features
- **Position Classification**: 5 classes (PG, SG, SF, PF, C) instead of player identification
- **One-hot Output**: Proper threshold function implementation
- **Team Balance**: Ensures one player per position for balanced teams
- **Interactive Interface**: Streamlit frontend with configurable parameters

## 🏗️ Architecture

```
Input Layer (13 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output Layer (5 neurons, Softmax)
    ↓
Threshold Function → One-hot Prediction
```

## 📊 Features Used

**Player Characteristics (13 features):**
- Points per game (pts)
- Rebounds per game (reb)
- Assists per game (ast)
- Net rating
- True shooting percentage (ts_pct)
- Usage percentage (usg_pct)
- Offensive rebound percentage (oreb_pct)
- Defensive rebound percentage (dreb_pct)
- Assist percentage (ast_pct)
- Player height
- Player weight
- Age
- Composite performance score

## 🎯 Team Selection Process

1. **Data Preprocessing**: Filter 2018-2023 seasons, select top 100 players
2. **Position Classification**: Classify players into 5 basketball positions
3. **Neural Network Training**: Train MLP on position classification task
4. **Threshold Application**: Convert softmax probabilities to one-hot predictions
5. **Team Construction**: Select one player per position for balanced team

## 🛠️ Installation & Usage

### Local Development
```bash
# Clone repository
git clone [your-repo-url]
cd ANN

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Upload `all_seasons.csv.xlsx` dataset
4. Deploy automatically

## 📁 Project Structure

```
ANN/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── all_seasons.csv.xlsx     # NBA dataset (2018-2023)
└── README.md                # Project documentation
```

## 🧠 Neural Network Details

### Activation Functions
- **Hidden Layers**: ReLU (Rectified Linear Unit)
- **Output Layer**: Softmax (probability distribution)

### Training Process
- **Loss Function**: Cross-entropy loss
- **Optimization**: Mini-batch gradient descent
- **Weight Initialization**: Xavier/He initialization
- **Batch Size**: Configurable (default: 16)
- **Learning Rate**: Configurable (default: 0.001)

### Position Classification Logic
```python
# Position assignment based on height and stats
if height < 190 and assists > 5:     # Point Guard
elif height < 195 and assists > 3:   # Shooting Guard
elif height < 205:                   # Small Forward
elif height < 210 and rebounds > 6:  # Power Forward
else:                                # Center
```

## 📈 Results & Analysis

The neural network successfully:
- Classifies players into distinct basketball positions
- Selects balanced teams with complementary skills
- Provides confidence scores for position predictions
- Visualizes team statistics and balance metrics

## 🔧 Configuration Options

**Streamlit Interface:**
- Hidden layer sizes (32-256 neurons)
- Learning rate (0.0001-0.01)
- Training epochs (50-500)
- Real-time training progress
- Interactive visualizations

## 📊 Visualizations

- Training loss curves
- Team balance radar charts
- Player statistics comparisons
- Position prediction confidence
- Team composition analysis

## 🚀 Deployment

This app is designed for easy deployment on **Streamlit Community Cloud**:

1. **GitHub Repository**: Push all files to GitHub
2. **Streamlit Cloud**: Connect repository
3. **Dataset Upload**: Upload NBA dataset file
4. **Automatic Deployment**: App deploys automatically

## 📝 Technical Report

For the complete technical documentation including:
- Problem statement analysis
- Detailed algorithm explanation
- Performance analysis and findings
- Academic references

Please refer to the separate technical report document.

## 🎓 Academic Context

**Course**: CST-435 - Artificial Neural Networks
**Assignment**: Build and deploy a deep ANN for practical application
**Focus**: MLP implementation with forward/backward propagation

## 🔗 Links

- **Live Demo**: [https://4rvzy3vjhzfyfhmrp2dsqx.streamlit.app/]
---

*This project demonstrates a complete artificial neural network implementation for sports analytics, showcasing the power of deep learning in team optimization and player classification.*