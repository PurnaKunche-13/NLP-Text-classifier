import pickle
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_models():
    with (BASE_DIR / "trained_model_text_topic.pkl").open("rb") as f:
        topic_model = pickle.load(f)

    with (BASE_DIR / "sentiment.pkl").open("rb") as f:
        sentiment_model = pickle.load(f)

    return topic_model, sentiment_model


def analyze_text(text):
    topic_model, sentiment_model = load_models()
    return {
        "topic": topic_model.predict([text])[0],
        "sentiment": sentiment_model.predict([text])[0],
        "characters": len(text),
        "words": len(text.split()),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("text", "").strip()

        if not user_input:
            error = "Please enter some text."
        else:
            try:
                result = analyze_text(user_input)
            except Exception as exc:
                error = (
                    f"Error: {exc}. Your saved model may not include a text "
                    "vectorizer such as TfidfVectorizer or CountVectorizer."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
        user_input=user_input,
    )


if __name__ == "__main__":
    app.run(debug=True)
