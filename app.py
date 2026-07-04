import pickle
import html

import streamlit as st

st.set_page_config(
    page_title="Topic Analysis",
    page_icon="TA",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 30rem),
                linear-gradient(135deg, #f7fbff 0%, #edf4f8 52%, #ffffff 100%);
            color: #172033;
        }

        .block-container {
            max-width: 820px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1 {
            color: #102033;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }

        .app-subtitle {
            color: #536173;
            font-size: 1.08rem;
            margin-bottom: 1.8rem;
        }

        .stTextArea label {
            color: #243247;
            font-weight: 700;
        }

        .stTextArea textarea {
            border: 1px solid #c8d6e3;
            border-radius: 14px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            font-size: 1rem;
            line-height: 1.55;
        }

        .stTextArea textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18);
        }

        .stButton > button {
            width: 100%;
            border: 0;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #16a34a);
            color: #ffffff;
            font-weight: 800;
            padding: 0.85rem 1.1rem;
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.22);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(37, 99, 235, 0.28);
            color: #ffffff;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.38);
            border-radius: 14px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin-top: 1rem;
            padding: 1rem 1.1rem;
        }

        .result-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-value {
            color: #142033;
            font-size: 1.3rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained models
with open("trained_model_text_topic.pkl", "rb") as file:
    topic_model = pickle.load(file)
with open("sentiment.pkl", "rb") as file:
    sentiment_model = pickle.load(file)

st.title("Text Topic Analysis")
st.markdown(
    '<p class="app-subtitle">Enter text below to predict its topic and sentiment.</p>',
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Enter text",
    height=220,
    placeholder="Type or paste your text here...",
)

if st.button("Predict"):
    if user_input.strip():
        try:
            topic_prediction = topic_model.predict([user_input])[0]
            sentiment_prediction = sentiment_model.predict([user_input])[0]
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Topic</div>
                    <div class="result-value">{html.escape(str(topic_prediction))}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Predicted Sentiment</div>
                    <div class="result-value">{html.escape(str(sentiment_prediction))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Your saved model may require a vectorizer, such as TfidfVectorizer or CountVectorizer.")
    else:
        st.warning("Please enter some text.")
