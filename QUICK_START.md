# ⚡ Quick Start Guide

Get your AI/ML Showcase running in 5 minutes!

---

## 🚀 Run Locally (5 Steps)

### 1️⃣ Install Python

Make sure you have Python 3.8+ installed:
```bash
python --version
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

⏱️ **Time:** 2-3 minutes

### 3️⃣ Download NLTK Data (for Sentiment Analysis)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

⏱️ **Time:** 30 seconds

### 4️⃣ Run the App

```bash
streamlit run Home.py
```

⏱️ **Time:** 10 seconds

### 5️⃣ Open in Browser

Your default browser will automatically open to:
```
http://localhost:8501
```

🎉 **Done!** Your showcase is running!

---

## 🌐 Deploy to Web (3 Steps)

### 1️⃣ Push to GitHub

```bash
git init
git add .
git commit -m "AI/ML Showcase"
git remote add origin https://github.com/YOUR-USERNAME/ai-ml-showcase.git
git push -u origin main
```

### 2️⃣ Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Main file: `Home.py`
6. Click "Deploy"

### 3️⃣ Share Your URL!

Your app will be live at:
```
https://your-app-name.streamlit.app
```

⏱️ **Time:** 5 minutes total

---

## 🎮 Navigate the Showcase

### Home Page
- Overview of all 5 projects
- Performance metrics
- Technology stack

### Project Pages (Sidebar)
1. 🎯 **Perceptron** - Furniture Placement
2. 🏀 **Deep ANN** - NBA Team Selection
3. 🖼️ **CNN** - Image Recognition
4. 💬 **NLP** - Sentiment Analysis
5. 🎭 **DCGAN** - Face Generation

### Each Project Has 4 Tabs:
- 📖 **Overview:** Theory and explanation
- 🎮 **Try It Out:** Interactive demo
- 📊 **Results:** Performance analysis
- 💻 **Code:** Implementation details

---

## 🛠️ Troubleshooting

### Issue: Module not found

```bash
pip install -r requirements.txt
```

### Issue: Port already in use

```bash
streamlit run Home.py --server.port 8502
```

### Issue: NLTK data not found

```bash
python -c "import nltk; nltk.download('all')"
```

### Issue: Can't see emoji in filenames

Windows users: Emojis work in browser, but may show as `?` in terminal. This is normal!

---

## 📁 File Structure

```
SHOWCASE/
├── Home.py                    ← START HERE
├── requirements.txt           ← Dependencies
├── README.md                  ← Full documentation
├── pages/                     ← Auto-loaded by Streamlit
│   ├── 1_🎯_Perceptron.py
│   ├── 2_🏀_NBA_Team_Selection.py
│   ├── 3_🖼️_CNN_Image_Recognition.py
│   ├── 4_💬_Sentiment_Analysis.py
│   └── 5_🎭_DCGAN_Face_Generation.py
└── all_seasons.csv.xlsx       ← NBA dataset
```

---

## 💡 Tips

### Speed Up Loading
- Comment out TensorFlow in requirements.txt if you don't need CNN training
- Models load on demand, not at startup

### Customize Theme
- Edit `.streamlit/config.toml`
- Change colors, fonts, etc.

### Add Your Info
- Edit `Home.py` to add your name/links
- Update project descriptions

---

## 🎯 What's Included

✅ 5 Complete ML Projects
✅ Interactive Demos
✅ Educational Content
✅ Professional UI
✅ Deployment Ready
✅ Mobile Responsive
✅ Production Code

---

## 📚 Learn More

- **Full Documentation:** [README.md](README.md)
- **Deployment Guide:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)

---

## 🆘 Need Help?

1. Check [README.md](README.md) for detailed docs
2. Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment help
3. Visit [Streamlit Community](https://discuss.streamlit.io)

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed
- [ ] NLTK data downloaded
- [ ] App runs on localhost:8501
- [ ] All 5 projects load
- [ ] Sidebar navigation works
- [ ] Ready to deploy!

---

**🎉 Enjoy your AI/ML Showcase!**

*Built with ❤️ using Streamlit, TensorFlow, PyTorch, and scikit-learn*
