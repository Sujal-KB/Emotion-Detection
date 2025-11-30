import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')
@st.cache_resource
def load_models():
    model_path=hf_hub_download(
        repo_id='SujalKB/emotion_detection',
        filename='model.keras'
    )
    model=load_model(model_path)

    tokenizer_path=hf_hub_download(
        repo_id='SujalKB/emotion_detection',
        filename='tokenizer.pkl'
    )
    tokenizer=joblib.load(tokenizer_path)

    encoder_path=hf_hub_download(
        repo_id='SujalKB/emotion_detection',
        filename='encoder.pkl'
    )
    encoder=joblib.load(encoder_path)
    return model,tokenizer,encoder

model,tokenizer,encoder=load_models()

def predict_fn(msg):
    msg_seq=tokenizer.texts_to_sequences(msg)
    msg_pad=pad_sequences(msg_seq,padding='pre',maxlen=300)
    pred=model.predict(msg_pad)
    return pred

explainer=LimeTextExplainer(class_names=encoder.classes_)

st.set_page_config(page_title="Emotion Detection")
st.title("Text based Emotion Detector")

st.markdown('###### Detects Emotions like Joy, Sad, Angry, Fear, Surprise')
with st.form("form_input"):
    msg = st.text_area("Text/Message:", placeholder='Enter the text here')
    bt = st.form_submit_button("Detect", type='primary')
    if bt:
        with st.spinner("Analysing..."):
            msg_seq = tokenizer.texts_to_sequences([msg])
            msg_pad = pad_sequences(msg_seq, maxlen=300, padding='pre')

            probs = model.predict(msg_pad)  # probabilities for all classes
            pred_idx = np.argmax(probs, axis=1)[0]  # integer index of best class
            pred_class = encoder.classes_[pred_idx]  # class name string

            st.session_state["last_text"] = msg
            st.session_state["last_pred_idx"] = pred_idx   # store index for LIME

            st.success(f"Predicted class name : {pred_class.upper()}")

        with st.spinner("Generating Explanation...", show_time=True):
            exp = explainer.explain_instance(
                st.session_state["last_text"],
                predict_fn,
                num_features=10,
                labels=[st.session_state["last_pred_idx"]]   # pass INT index
            )
            st.session_state["last_exp"] = exp

            html = exp.as_html()
            html_with_bg = f"""
                <div style="background-color:white;padding:10px;color:black;">{html}</div>
            """
            components.html(html_with_bg, height=450,scrolling=True)
