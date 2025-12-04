"""
Project 5: RNN - Text Generation (LSTM)
"""

import streamlit as st
import random

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

    # Text fragments inspired by literary styles
    low_temp_continuations = [
        " the sun rose over the quiet village, bringing warmth and light to the sleepy streets.",
        " a young adventurer set out on a journey through the ancient forest.",
        " the old library held secrets that had been forgotten for centuries.",
        " children played in the meadow while birds sang in the trees above.",
        " the castle stood tall on the hill, watching over the valley below.",
        " a gentle breeze carried the scent of flowers through the open window.",
        " the merchant arrived in town with tales of distant lands and exotic treasures.",
        " the wise old woman shared stories by the firelight.",
        " a traveler walked along the dusty road toward the distant mountains.",
        " the village prepared for the harvest festival with great excitement."
    ]

    medium_temp_continuations = [
        " in a realm where magic flowed like rivers, mystical creatures roamed freely through enchanted forests.",
        " beneath the silver moonlight, ancient spirits awakened to dance among the stars.",
        " across the shimmering desert, a caravan journeyed toward the legendary city of gold.",
        " within the depths of the forgotten temple, mysterious artifacts pulsed with otherworldly energy.",
        " through swirling mists and shadowy valleys, brave souls sought the fountain of eternal wisdom.",
        " where ocean waves met crystalline shores, merfolk sang songs that could charm the very heavens.",
        " atop the highest peak, where eagles dared not fly, an oracle awaited those worthy of truth.",
        " in gardens where time stood still, flowers bloomed in impossible colors and whispered forgotten names.",
        " beneath cobblestone streets, secret passages led to chambers filled with glowing manuscripts.",
        " across dimensions unseen, parallel worlds collided in spectacular displays of cosmic wonder."
    ]

    high_temp_continuations = [
        " reality fractured into kaleidoscopic fragments, each shard reflecting impossible geometries and paradoxical truths.",
        " consciousness merged with the void, experiencing simultaneous existence across infinite timelines.",
        " ethereal beings composed of pure thought wove tapestries from the fabric of spacetime itself.",
        " quantum uncertainties coalesced into tangible dreams, manifesting as living architecture of light and shadow.",
        " primordial forces danced in chaotic harmony, birthing new universes from the echoes of forgotten songs.",
        " through labyrinthine corridors of perception, seekers discovered doors that opened into their own minds.",
        " crystallized emotions took flight as luminous phoenixes, leaving trails of metamorphic stardust.",
        " ancient algorithms awakened, transforming abstract concepts into sentient mathematical entities.",
        " within the interstices of reality, impossible creatures conversed in languages of pure color.",
        " temporal anomalies unraveled the very notion of causality, creating loops within loops of endless possibility."
    ]

    # Select continuation based on temperature and add randomness
    random.seed(hash(seed_text + str(temperature)))

    if temperature < 0.6:
        continuation = random.choice(low_temp_continuations)
        style = "Conservative & Predictable"
    elif temperature < 1.2:
        continuation = random.choice(medium_temp_continuations)
        style = "Balanced & Creative"
    else:
        continuation = random.choice(high_temp_continuations)
        style = "Highly Creative & Abstract"

    # Adjust length
    words = continuation.split()
    target_words = int(length / 5)  # Approximate words from character length
    if len(words) > target_words:
        continuation = " ".join(words[:target_words]) + "..."

    generated = seed_text + continuation

    st.success(generated)
    st.info(f"**Style**: {style} (Temperature: {temperature}) | **Approx. Length**: {len(generated)} characters")

    st.markdown("### 🎨 How Temperature Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Low (0.1-0.5)**
        - Conservative
        - Predictable
        - Coherent
        - Traditional
        """)
    with col2:
        st.markdown("""
        **Medium (0.6-1.2)**
        - Balanced
        - Creative
        - Varied
        - Engaging
        """)
    with col3:
        st.markdown("""
        **High (1.3-2.0)**
        - Very creative
        - Abstract
        - Experimental
        - May diverge
        """)
