"""
Project 5: RNN - Text Generation (LSTM)
"""

import streamlit as st

st.set_page_config(page_title="RNN Text Generation", page_icon="📝", layout="wide")
st.title("📝 RNN: Text Generation (LSTM)")

with st.expander("📖 Show Project Overview", expanded=False):
    st.markdown("""
    ### Purpose
    Generate coherent text using a Recurrent Neural Network (LSTM) trained on "Alice in Wonderland".

    ### How It Works
    1. **Data Preparation**: Tokenize text, create sequences, build vocabulary
    2. **LSTM Architecture**: Embedding → LSTM(128) → Dense(vocab_size)
    3. **Training**: Learn next-word prediction patterns
    4. **Generation**: Use seed text + temperature sampling to generate new text

    ### Results
    - Architecture: Embedding → LSTM(128 units) → Dense(softmax)
    - Training: Categorical cross-entropy, 50 epochs
    - Temperature: Controls randomness (low=conservative, high=creative)
    - Dataset: "Alice in Wonderland" (~140KB text)
    """)

st.markdown("---")

st.markdown("### 🎮 Text Generation Demo")
st.info("💡 This demo shows the concept. Full model requires TensorFlow training on complete text corpus.")

col1, col2 = st.columns(2)
with col1:
    seed_text = st.text_input("Seed Text:", value="Once upon a time")
    temperature = st.slider("Temperature (randomness)", 0.1, 2.0, 1.0, 0.1)
with col2:
    length = st.slider("Generated Length", 10, 100, 50)

if st.button("🚀 Generate Text", type="primary"):
    st.markdown("### 📊 Generated Text")
    if temperature < 0.6:
        generated = seed_text + " there was a little girl who wandered through the forest..."
    elif temperature < 1.2:
        generated = seed_text + " in a magical kingdom far away, mystical creatures danced..."
    else:
        generated = seed_text + " beneath twisted moonbeams, enigmatic shadows whispered secrets..."

    st.success(generated)
    st.info("Note: This is a demonstration. Train full LSTM model for actual text generation.")

    st.markdown("### How Temperature Works")
    st.markdown("""
    - **Low (0.1-0.5)**: Conservative, predictable text
    - **Medium (0.6-1.2)**: Balanced creativity and coherence
    - **High (1.3-2.0)**: Creative but may lose coherence
    """)
