# 🤖 AI/ML Projects Showcase

A unified Streamlit application showcasing 5 comprehensive machine learning projects from neural networks to deep learning.

**Author:** Aime Serge Tuyishime
**Course:** CST 435 - Neural Networks & Deep Learning
**Institution:** Grand Canyon University

---

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🎯 Overview

This showcase presents **5 machine learning projects** in a single, unified Streamlit web application. Each project demonstrates different aspects of AI/ML, from classical algorithms to modern deep learning architectures.

**Live Demo:** Deploy this on Streamlit Cloud to get your live URL!

---

## 🚀 Projects

### 1. 🎯 Perceptron - Furniture Placement Optimization

**Algorithm:** Perceptron (Classical Neural Network)

- Binary classification for optimal furniture placement
- Feature engineering from spatial distances
- Real-time decision boundary visualization
- Interactive room configuration

**Key Metrics:**
- Convergence: ~20-50 epochs
- Accuracy: 95%+
- Training samples: 200

### 2. 🏀 Deep ANN - NBA Team Selection

**Algorithm:** Multi-Layer Perceptron (Deep Learning)

- 5-player team selection from 100 NBA players
- Forward/backward propagation implementation
- Position-based classification (PG, SG, SF, PF, C)
- Real-time training visualization

**Key Metrics:**
- Architecture: 13 → 128 → 64 → 32 → 5
- Training accuracy: 80%+
- Dataset: NBA stats 2018-2023
- **Deployed:** ✅ [View Live](https://4rvzy3vjhzfyfhmrp2dsqx.streamlit.app/)

### 3. 🖼️ CNN - Image Recognition (CIFAR-10)

**Algorithm:** Convolutional Neural Network

- Multi-class image classification (10 categories)
- 3-layer CNN with max pooling
- Real-time image upload and prediction
- 60,000 training images

**Key Metrics:**
- Test accuracy: 69.33%
- Parameters: 282,250
- Training: 50 epochs
- Classes: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

### 4. 💬 NLP - Sentiment Analysis (IMDB)

**Algorithm:** TF-IDF + Logistic Regression

- Binary sentiment classification (positive/negative)
- Text preprocessing pipeline
- Feature importance analysis
- Real-time review analysis

**Key Metrics:**
- Test accuracy: 89.45%
- Precision/Recall: ~89%
- Dataset: 50,000 IMDB reviews
- **Deployed:** ✅ [View Live](https://nlp-sentiment-analysis-qc0c.onrender.com)

### 5. 🎭 DCGAN - Face Generation

**Algorithm:** Deep Convolutional GAN

- Generate realistic 64×64 RGB faces
- Trained on 202K celebrity faces (CelebA)
- Adversarial training (Generator vs Discriminator)
- Latent space exploration

**Key Metrics:**
- Parameters: 15.5M
- Training: 25 epochs (5 hours on GPU)
- Resolution: 64×64 RGB
- **Deployed:** ✅ [View Live](https://huggingface.co/spaces/nshutimchristian/DCGAN)

---

## ✨ Features

### Interactive Demos
- Real-time predictions and visualizations
- Adjustable hyperparameters
- Upload custom data (images, text)
- Live training monitoring

### Educational Content
- Detailed algorithm explanations
- Step-by-step implementation guides
- Architecture diagrams
- Performance analysis

### Professional UI
- Modern, responsive design
- Multi-page navigation
- Interactive visualizations
- Mobile-friendly

### Deployment Ready
- Single-command deployment
- Optimized for Streamlit Cloud
- All dependencies included
- Production-ready code

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd SHOWCASE
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (for Sentiment Analysis)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Step 5: Verify Installation

```bash
streamlit --version
```

---

## 🎮 Usage

### Running Locally

```bash
streamlit run Home.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation

1. **Home Page:** Overview of all 5 projects
2. **Sidebar:** Navigate between projects
3. **Project Pages:** Interactive demos with multiple tabs
   - 📖 Overview: Theory and explanation
   - 🎮 Try It Out: Interactive demos
   - 📊 Results: Performance metrics
   - 💻 Code: Implementation details

### Tips for Best Experience

- **CNN Project:** TensorFlow is optional (large dependency). Uncomment in requirements.txt if needed.
- **ANN Project:** Requires `all_seasons.csv.xlsx` file (included)
- **Perceptron:** Fully self-contained, no external data needed
- **Sentiment Analysis:** Works with pre-trained models
- **DCGAN:** Links to deployed HuggingFace demo

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (FREE)

#### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI/ML Showcase"
git remote add origin <your-github-repo-url>
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `Home.py`
6. Click "Deploy"

#### Step 3: Done! 🎉

Your app will be live at: `https://<your-app-name>.streamlit.app`

### Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

---

## 🛠️ Technologies

### Frameworks & Libraries

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **NumPy** | Numerical computing |
| **Pandas** | Data manipulation |
| **Scikit-learn** | Machine learning algorithms |
| **TensorFlow/Keras** | Deep learning (CNN) |
| **PyTorch** | Deep learning (GAN) |
| **NLTK** | Natural language processing |
| **Matplotlib/Plotly** | Data visualization |

### Deployment Platforms

- **Streamlit Cloud:** Main app hosting (FREE)
- **HuggingFace Spaces:** DCGAN demo
- **Render:** Sentiment analysis API

---

## 📁 Project Structure

```
SHOWCASE/
│
├── Home.py                              # Main landing page
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── all_seasons.csv.xlsx                # NBA dataset (for Project 2)
│
├── pages/                              # Streamlit pages (auto-detected)
│   ├── 1_🎯_Perceptron.py              # Project 1: Perceptron
│   ├── 2_🏀_NBA_Team_Selection.py      # Project 2: Deep ANN
│   ├── 3_🖼️_CNN_Image_Recognition.py   # Project 3: CNN
│   ├── 4_💬_Sentiment_Analysis.py      # Project 4: NLP
│   └── 5_🎭_DCGAN_Face_Generation.py   # Project 5: GAN
│
├── .streamlit/                         # Streamlit configuration
│   └── config.toml                    # Theme and settings
│
├── PERCEPTRON PROJECT 1/               # Original project files
│   └── perceptron_code.py
│
├── ANN PROJECT 2/                      # Original project files
│   ├── app.py
│   ├── all_seasons.csv.xlsx
│   └── README.md
│
├── CNN_Image_Recognition PROJECT 3/    # Original project files
│   └── CNN_Image_Recognition.ipynb
│
├── PROJECT 4 Sentiment_Analysis_Assignment/
│   └── Sentiment_Analysis_Assignment.ipynb
│
└── DCGAN PROJECT 6/                    # Original project files
    └── Face_Generation_DCGAN_Colab_FIXED (1).ipynb
```

---

## 📊 Performance Summary

| Project | Algorithm | Accuracy | Parameters | Dataset Size |
|---------|-----------|----------|------------|--------------|
| Perceptron | Perceptron | 95%+ | 5 | 200 samples |
| NBA ANN | Deep MLP | 80%+ | ~280K | 100 players |
| CNN | ConvNet | 69.33% | 282K | 60K images |
| Sentiment | LogReg + TF-IDF | 89.45% | ~5K features | 50K reviews |
| DCGAN | GAN | N/A (generative) | 15.5M | 202K faces |

---

## 🎓 Educational Value

This showcase demonstrates:

1. **Classical ML:** Perceptron, Logistic Regression
2. **Deep Learning:** MLPs, CNNs
3. **Generative Models:** GANs
4. **NLP:** Text preprocessing, TF-IDF
5. **Computer Vision:** Image classification, generation
6. **Model Deployment:** Production-ready applications

Perfect for:
- Course portfolios
- Job applications
- Learning resources
- Teaching materials

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`
```bash
Solution: pip install -r requirements.txt
```

**Issue:** NLTK data not found
```bash
Solution: python -c "import nltk; nltk.download('all')"
```

**Issue:** TensorFlow installation fails
```bash
Solution: TensorFlow is optional. Comment it out in requirements.txt
```

**Issue:** Out of memory during CNN training
```bash
Solution: Reduce batch size or use Google Colab
```

---

## 📝 License

This project is created for educational purposes as part of CST 435 coursework at Grand Canyon University.

**Datasets:**
- CIFAR-10: Public domain
- IMDB Reviews: Public domain
- CelebA: Research use
- NBA Stats: Public data

---

## 👤 Author

**Aime Serge Tuyishime**

- Course: CST 435 - Neural Networks & Deep Learning
- Institution: Grand Canyon University
- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]

---

## 🙏 Acknowledgments

- **TensorFlow/Keras Team** - Deep learning frameworks
- **PyTorch Team** - GAN implementation
- **Streamlit Team** - Amazing web framework
- **scikit-learn** - Machine learning tools
- **CIFAR-10, IMDB, CelebA** - Dataset providers
- **Grand Canyon University** - Educational support

---

## 📞 Support

For questions or issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review individual project README files
3. Contact course instructor
4. Open an issue on GitHub

---

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone <repo-url>
cd SHOWCASE
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run locally
streamlit run Home.py

# Deploy (after pushing to GitHub)
# Go to share.streamlit.io and connect your repo
```

---

## 📈 Future Enhancements

Potential improvements:
- [ ] Add more projects (RNN, Transformer, etc.)
- [ ] Enable model training from UI
- [ ] Add model comparison tools
- [ ] Implement user authentication
- [ ] Add data upload/download features
- [ ] Create API endpoints
- [ ] Add model performance dashboards

---

**⭐ If you found this helpful, please star the repository!**

---

*Last Updated: December 2024*
