"""
AI/ML Projects Showcase
========================
A unified portfolio showcasing 6 machine learning projects

Authors: Aime Serge Tuyishime & Christian Nshuti Manzi
Course: CST 435 - Neural Networks & Deep Learning
Institution: Grand Canyon University
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI/ML Projects Showcase",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, cool styling
st.markdown("""
<style>
    /* Base font sizes */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* Animated gradient background for header */
    .main-header {
        text-align: center;
        padding: 3rem 0 2.5rem 0;
        margin-bottom: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 3.2rem !important;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .main-header p {
        font-size: 1.2rem !important;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 500;
    }

    /* Modern project card with gradient border */
    .project-card {
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #667eea 0%, #764ba2 100%) border-box;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        border: 3px solid transparent;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        height: 100%;
        min-height: 240px;
        position: relative;
        overflow: hidden;
    }

    .project-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
        border-radius: 16px;
    }

    .project-card:hover::before {
        opacity: 1;
    }

    .project-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }

    .project-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        display: inline-block;
        transition: transform 0.3s ease;
    }

    .project-card:hover .project-icon {
        transform: scale(1.2) rotate(5deg);
    }

    .project-number {
        display: inline-block;
        font-size: 0.75rem !important;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        letter-spacing: 0.5px;
    }

    .project-title {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.4rem;
        position: relative;
        z-index: 1;
    }

    .project-subtitle {
        font-size: 1rem !important;
        color: #6b7280;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }

    /* Gradient button styling */
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.6rem 1.8rem !important;
        border-radius: 25px !important;
        font-weight: 600;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }

    /* Modern description section */
    .description-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-top: 3rem;
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.1);
    }

    .description-section h3 {
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .description-section p {
        font-size: 1.1rem !important;
        color: #4b5563;
        line-height: 1.8;
        max-width: 900px;
        margin: 0 auto;
    }

    /* Enhanced sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
    }

    section[data-testid="stSidebar"] nav a {
        font-size: 1.2rem !important;
        padding: 0.9rem 1.1rem !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    section[data-testid="stSidebar"] nav a:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }

    /* Modern footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
        color: #4b5563;
        font-size: 1rem !important;
    }

    .footer strong {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Header with animated gradient
st.markdown("""
<div class="main-header">
    <h1>🚀 AI/ML Projects Showcase</h1>
    <p>Neural Networks & Deep Learning Portfolio</p>
</div>
""", unsafe_allow_html=True)

# Quick stats section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Projects", "6", delta="Complete")
with col2:
    st.metric("Neural Nets", "5+", delta="Trained")
with col3:
    st.metric("Frameworks", "3", delta="TF, PyTorch, SK")
with col4:
    st.metric("Accuracy", "85%+", delta="Average")

st.markdown("<br>", unsafe_allow_html=True)

# Project Cards - 3x2 Grid Layout
# Row 1: Projects 1, 2, 3
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">📊</div>
        <div class="project-number">PROJECT 1</div>
        <div class="project-title">Perceptron</div>
        <div class="project-subtitle">Furniture Placement Optimization</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p1", type="primary"):
        st.switch_page("pages/1_🎯_Perceptron.py")

with col2:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">🏀</div>
        <div class="project-number">PROJECT 2</div>
        <div class="project-title">Deep ANN</div>
        <div class="project-subtitle">NBA Team Selection</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p2", type="primary"):
        st.switch_page("pages/2_🏀_NBA_Team_Selection.py")

with col3:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">🖼️</div>
        <div class="project-number">PROJECT 3</div>
        <div class="project-title">CNN</div>
        <div class="project-subtitle">Image Classification</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p3", type="primary"):
        st.switch_page("pages/3_🖼️_CNN_Image_Recognition.py")

# Spacing between rows
st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Projects 4, 5, 6
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">💬</div>
        <div class="project-number">PROJECT 4</div>
        <div class="project-title">NLP</div>
        <div class="project-subtitle">Sentiment Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p4", type="primary"):
        st.switch_page("pages/4_💬_Sentiment_Analysis.py")

with col2:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">📝</div>
        <div class="project-number">PROJECT 5</div>
        <div class="project-title">RNN</div>
        <div class="project-subtitle">Text Generation</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p5", type="primary"):
        st.switch_page("pages/5_📝_RNN_Text_Generation.py")

with col3:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon">🎭</div>
        <div class="project-number">PROJECT 6</div>
        <div class="project-title">DCGAN</div>
        <div class="project-subtitle">Face Generation</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Project", key="p6", type="primary"):
        st.switch_page("pages/6_🎭_DCGAN_Face_Generation.py")

# Description Section
st.markdown("""
<div class="description-section">
    <h3>About This Showcase</h3>
    <p>
        This portfolio demonstrates comprehensive machine learning and deep learning expertise through
        6 projects spanning computer vision, natural language processing, and generative modeling.
        Each project showcases end-to-end implementation from data preprocessing to model deployment,
        utilizing industry-standard frameworks including TensorFlow, PyTorch, and scikit-learn.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <strong>Aime Serge Tuyishime</strong> & <strong>Christian Nshuti Manzi</strong><br>
    CST 435 - Neural Networks & Deep Learning | Grand Canyon University
</div>
""", unsafe_allow_html=True)
