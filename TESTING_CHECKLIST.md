# ✅ Testing Checklist

Use this checklist to verify everything works before deploying.

---

## 🧪 Local Testing

### **Step 1: Installation Test**

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Virtual environment created (optional but recommended)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] No installation errors

### **Step 2: App Launch Test**

- [ ] App starts without errors (`streamlit run Home.py`)
- [ ] Opens in browser automatically
- [ ] URL is `http://localhost:8501`
- [ ] No error messages in terminal

### **Step 3: Home Page Test**

- [ ] Page loads completely
- [ ] Header displays correctly
- [ ] All 5 project cards visible
- [ ] Stats section shows correctly
- [ ] Technologies section displays
- [ ] Dataset table shows
- [ ] No broken images
- [ ] Footer displays

### **Step 4: Navigation Test**

- [ ] Sidebar menu appears
- [ ] All 5 projects listed in sidebar
- [ ] Clicking each project loads correct page
- [ ] "Back to Home" link works
- [ ] Emoji icons display (or show as boxes on Windows - OK)

### **Step 5: Project 1 - Perceptron Test**

- [ ] Page loads without errors
- [ ] All 4 tabs present (Overview, Demo, Results, Code)
- [ ] Overview tab shows content
- [ ] Demo tab has sliders
- [ ] "Train Perceptron" button works
- [ ] Training produces visualizations
- [ ] No console errors

### **Step 6: Project 2 - NBA ANN Test**

- [ ] Page loads without errors
- [ ] All 4 tabs present
- [ ] NBA dataset loads (`all_seasons.csv.xlsx` found)
- [ ] Training configuration sliders work
- [ ] "Train Neural Network" button works
- [ ] Progress bar shows during training
- [ ] Results display correctly
- [ ] No missing data errors

### **Step 7: Project 3 - CNN Test**

- [ ] Page loads without errors
- [ ] All 4 tabs present
- [ ] "Load Pre-trained Model" button works
- [ ] File upload widget appears
- [ ] "Load Random CIFAR-10 Images" works
- [ ] Images display correctly
- [ ] No TensorFlow errors (if installed)

**Note:** If TensorFlow not installed, demo will show warning but page should still load.

### **Step 8: Project 4 - Sentiment Analysis Test**

- [ ] Page loads without errors
- [ ] All 4 tabs present
- [ ] "Load Sentiment Model" button works
- [ ] Text area appears for input
- [ ] Example reviews selectable
- [ ] "Analyze Sentiment" works
- [ ] Results show positive/negative
- [ ] Confidence scores display

### **Step 9: Project 5 - DCGAN Test**

- [ ] Page loads without errors
- [ ] All 4 tabs present
- [ ] HuggingFace link button displays
- [ ] Link opens in new tab
- [ ] Architecture explanation shows
- [ ] Code snippets display correctly
- [ ] No broken content

---

## 🌐 Deployment Testing

### **Step 1: GitHub Push Test**

- [ ] Repository created on GitHub
- [ ] All files pushed successfully
- [ ] `.gitignore` working (no `__pycache__`, `venv/`)
- [ ] README.md displays nicely on GitHub
- [ ] `all_seasons.csv.xlsx` included (1.7MB file)

### **Step 2: Streamlit Cloud Deployment Test**

- [ ] Logged into share.streamlit.io
- [ ] Repository connected
- [ ] App deployed successfully
- [ ] Build logs show no errors
- [ ] App status shows "Running"
- [ ] Public URL accessible

### **Step 3: Deployed App Test**

- [ ] URL works in browser
- [ ] Home page loads
- [ ] All 5 projects accessible
- [ ] Images/visualizations display
- [ ] No "File not found" errors
- [ ] No "Module not found" errors
- [ ] Sidebar navigation works
- [ ] External links work (HuggingFace, etc.)

### **Step 4: Mobile Test**

- [ ] Open URL on mobile device
- [ ] Page displays correctly
- [ ] Sidebar menu accessible
- [ ] Buttons are tappable
- [ ] Text is readable
- [ ] Images scale properly

### **Step 5: Performance Test**

- [ ] Page loads in <5 seconds
- [ ] No lag when navigating
- [ ] Buttons respond quickly
- [ ] No memory issues
- [ ] App doesn't crash

---

## 🐛 Common Issues & Solutions

### **Issue: `ModuleNotFoundError: No module named 'streamlit'`**
```bash
Solution: pip install -r requirements.txt
```

### **Issue: `FileNotFoundError: all_seasons.csv.xlsx`**
```bash
Solution: Make sure file is in SHOWCASE directory
Check with: ls all_seasons.csv.xlsx
```

### **Issue: NLTK data not found**
```bash
Solution: python -c "import nltk; nltk.download('all')"
```

### **Issue: Port 8501 already in use**
```bash
Solution: streamlit run Home.py --server.port 8502
```

### **Issue: TensorFlow import fails**
```bash
Solution: Comment out tensorflow in requirements.txt
CNN project will show warning but still work
```

### **Issue: Emoji showing as `?` in terminal**
```bash
Solution: This is normal on Windows. Emojis work fine in browser!
```

### **Issue: App crashes during training**
```bash
Solution: Reduce batch size or epochs in the demo
Check RAM usage (should have 4GB free)
```

### **Issue: Deployment fails on Streamlit Cloud**
```bash
Solutions:
1. Check build logs for error
2. Verify requirements.txt has correct versions
3. Make sure all files pushed to GitHub
4. Reboot app from dashboard
```

---

## ✅ Final Checklist Before Going Live

- [ ] All local tests passed
- [ ] Deployment successful
- [ ] Deployed app fully tested
- [ ] Mobile version works
- [ ] All links functional
- [ ] No console errors
- [ ] Performance acceptable
- [ ] Updated README with your live URL
- [ ] Added URL to resume/LinkedIn
- [ ] Shared with friends/professors

---

## 📊 Test Results Template

Fill this out after testing:

```
Test Date: _____________
Tester: _____________

Local Testing:
- Installation: ✅ / ❌
- App Launch: ✅ / ❌
- Navigation: ✅ / ❌
- Project 1: ✅ / ❌
- Project 2: ✅ / ❌
- Project 3: ✅ / ❌
- Project 4: ✅ / ❌
- Project 5: ✅ / ❌

Deployment Testing:
- GitHub Push: ✅ / ❌
- Streamlit Deploy: ✅ / ❌
- Live App: ✅ / ❌
- Mobile View: ✅ / ❌
- Performance: ✅ / ❌

Issues Found:
1. _______________________
2. _______________________
3. _______________________

Overall Result: PASS / FAIL

Notes:
_______________________________
_______________________________
```

---

## 🎯 Performance Benchmarks

**Expected Performance:**

| Metric | Target | Your Result |
|--------|--------|-------------|
| Initial Load Time | <5s | _____ |
| Page Switch Time | <2s | _____ |
| Demo Response Time | <3s | _____ |
| Memory Usage | <500MB | _____ |
| Mobile Load Time | <8s | _____ |

---

## 🆘 If Tests Fail

1. **Check requirements.txt**
   - All packages listed?
   - Correct versions?

2. **Check file structure**
   ```
   SHOWCASE/
   ├── Home.py  ← Must be here
   ├── pages/   ← Must contain 5 .py files
   ├── all_seasons.csv.xlsx  ← Must be here
   └── requirements.txt
   ```

3. **Check Python version**
   - Must be 3.8 or higher
   - Check with `python --version`

4. **Check disk space**
   - Need at least 2GB free

5. **Try fresh install**
   ```bash
   # Delete venv
   # Recreate venv
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run Home.py
   ```

---

## ✅ You're Done!

If all tests pass:
- ✅ Your app is production-ready
- ✅ Safe to share URL publicly
- ✅ Ready to add to resume
- ✅ Ready to demo to professors

**Congratulations! 🎉**

---

*Happy Testing! 🧪*
