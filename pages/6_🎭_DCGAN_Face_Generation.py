"""
Project 6: DCGAN - Face Generation
"""

import streamlit as st

st.set_page_config(page_title="DCGAN Face Generation", page_icon="🎭", layout="wide")
st.title("🎭 DCGAN: Human Face Generation")

with st.expander("📖 Show Project Overview", expanded=False):
    st.markdown("""
    ### Purpose
    Generate realistic 64×64 RGB human faces using a Deep Convolutional GAN trained on CelebA dataset (200K+ celebrity faces).

    ### How It Works
    1. **Generator**: Transforms random noise (100D) → 64×64×3 RGB image
       - Architecture: Dense → ConvTranspose2D layers with upsampling
       - Activation: ReLU in hidden layers, Tanh for output
    2. **Discriminator**: Classifies images as real or fake
       - Architecture: Conv2D layers with downsampling
       - Activation: LeakyReLU, Dropout for regularization
    3. **Training**: Adversarial learning - Generator vs Discriminator
       - Generator tries to fool Discriminator
       - Discriminator tries to detect fakes
       - Both improve through competition

    ### Results
    - Resolution: 64×64 RGB faces
    - Training: 25 epochs on Tesla T4 GPU (~5 hours)
    - Dataset: CelebA (202,599 celebrity faces)
    - Model: 15.5M parameters (Generator: 12.7M, Discriminator: 2.8M)
    """)

st.markdown("---")

st.markdown("### 🎭 Live Demo")
st.info("🚀 **Try the live face generator on HuggingFace!**")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.link_button("🎭 Launch Face Generator on HuggingFace",
                  "https://huggingface.co/spaces/nshutimchristian/DCGAN",
                  use_container_width=True, type="primary")

st.markdown("---")

st.markdown("### 🧠 How It Works")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Generator**
    - Input: Random noise (100 dimensions)
    - Process: Upsampling through transposed convolutions
    - Output: 64×64×3 RGB face image
    - Each random input creates unique face
    """)

with col2:
    st.markdown("""
    **Discriminator**
    - Input: 64×64×3 image (real or generated)
    - Process: Downsampling through convolutions
    - Output: Probability (0=fake, 1=real)
    - Guides Generator to improve quality
    """)

st.markdown("### 📊 Training Results")
st.markdown("""
- **Epoch 1-5**: Blurry shapes, basic colors
- **Epoch 5-10**: Face-like structures emerge
- **Epoch 10-15**: Clearer faces, better symmetry
- **Epoch 15-20**: Realistic textures, proper proportions
- **Epoch 20-25**: High-quality faces, fine details

**Stability**: Balanced G/D losses, no mode collapse
""")
