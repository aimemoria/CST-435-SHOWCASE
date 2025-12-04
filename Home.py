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

# Custom CSS for professional, clean styling
st.markdown("""
<style>
    /* Base font sizes */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 2rem 0;
        margin-bottom: 3rem;
    }

    .main-header h1 {
        font-size: 3.5rem !important;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.3rem !important;
        color: #6b7280;
    }

    /* Project card */
    .project-card {
        background: white;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
        min-height: 280px;
    }

    .project-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        border-color: #3b82f6;
    }

    .project-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    .project-number {
        font-size: 1rem !important;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }

    .project-title {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .project-subtitle {
        font-size: 1.1rem !important;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Button styling */
    .stButton > button {
        font-size: 1.15rem !important;
        padding: 0.7rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600;
        width: 100%;
    }

    /* Description section */
    .description-section {
        background: #f9fafb;
        padding: 2.5rem;
        border-radius: 12px;
        margin-top: 3rem;
        text-align: center;
    }

    .description-section h3 {
        font-size: 2rem !important;
        color: #1f2937;
        margin-bottom: 1rem;
    }

    .description-section p {
        font-size: 1.2rem !important;
        color: #4b5563;
        line-height: 1.8;
        max-width: 1000px;
        margin: 0 auto;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f9fafb;
    }

    section[data-testid="stSidebar"] nav a {
        font-size: 1.3rem !important;
        padding: 1rem 1.25rem !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem 0;
        margin-top: 4rem;
        border-top: 2px solid #e5e7eb;
        color: #4b5563;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>AI/ML Projects Showcase</h1>
    <p>Neural Networks & Deep Learning Portfolio</p>
</div>
""", unsafe_allow_html=True)

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
