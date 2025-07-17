import streamlit as st
import whisper
import torch

import time
import tempfile
import math
from pydub import AudioSegment

st.set_page_config(page_title="Whisper Transcription App", layout="centered")
st.title("🎤 Transcription Audio avec Whisper")

st.write(f"GPU disponible : {'✅' if torch.cuda.is_available() else '❌'}")
if torch.cuda.is_available():
    st.write(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
else:
    st.write("Aucun GPU détecté, utilisation du CPU.")

# Choix du modèle Whisper
model_size = st.selectbox(
    "Sélectionnez la taille du modèle Whisper :",
    ("tiny", "base", "small"), # "medium", "large"), too heavy for steramlit cloud
    index=0,
    help="Plus le modèle est grand, meilleure est la qualité, mais plus il est lent et consomme de ressources."
)

@st.cache_resource(show_spinner=True)
def load_whisper_model(selected_size):
    return whisper.load_model(selected_size).to("cuda" if torch.cuda.is_available() else "cpu")

model = load_whisper_model(model_size)

uploaded_file = st.file_uploader("Importer un fichier audio (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_audio_path = tmp_file.name

    # Parameters for chunking
    chunk_length_ms = 20 * 1000  # 20 seconds per chunk

    if st.button("Lancer la transcription"):

        st.info("Chargement et découpage de l'audio...")
        start_time = time.time()

        # Load audio with progress bar
        audio_loading_bar = st.progress(0, text="Chargement de l'audio...")
        audio = AudioSegment.from_file(tmp_audio_path)
        audio_loading_bar.progress(1.0, text="Audio chargé.")
        time.sleep(0.2)

        total_length_ms = len(audio)
        num_chunks = math.ceil(total_length_ms / chunk_length_ms)
        transcription = ""

        st.info("Transcription en cours...")
        progress_bar = st.progress(0, text="Transcription...")

        for i in range(num_chunks):
            start_ms = i * chunk_length_ms
            end_ms = min((i + 1) * chunk_length_ms, total_length_ms)
            chunk = audio[start_ms:end_ms]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                chunk_path = chunk_file.name
            result = model.transcribe(chunk_path, language="fr")
            transcription += result["text"] + "\n"
            progress_bar.progress((i + 1) / num_chunks, text=f"Chunk {i+1}/{num_chunks}")
            time.sleep(0.01)

        end_time = time.time()
        execution_time = end_time - start_time

        st.success("✅ Transcription terminée !")
        st.write(f"⏱️ Temps d'exécution : {execution_time:.2f} secondes")
        st.subheader("Transcription :")
        st.text_area("Texte transcrit", transcription, height=300)

        st.download_button(
            label="💾 Télécharger la transcription",
            data=transcription,
            file_name=f"transcription_{uploaded_file.name.rsplit('.',1)[0]}.txt",
            mime="text/plain"
        )