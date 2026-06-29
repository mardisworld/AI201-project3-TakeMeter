"""
app.py — PolyAnalyzer web interface.

Write your own response to the discussion thread and get its predicted
contribution type (analysis / op_response / response / question). Choose between
two models with the toggle:

  • Fine-tuned DistilBERT  — local model, returns a calibrated confidence per label
  • Zero-shot Groq         — llama-3.3-70b-versatile, returns a single hard choice
                             (no calibrated probability)

Run the pipeline first so the fine-tuned model exists:
    python preprocess.py --input ../Reddit-thread-on-American-presidency-v2.csv
    python train.py                 # writes ./model

For the Groq option, set an API key (a .env file is loaded automatically):
    GROQ_API_KEY=...

Then launch:
    pip install -r requirements.txt
    python app.py                   # opens http://127.0.0.1:7860
"""

import os

import gradio as gr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from labels import LABEL_MAP, ID_TO_LABEL, NUM_LABELS
from prompts import classification_system_prompt, classification_user_prompt
from baseline import parse_label

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_LENGTH = 256

# the prompt that grounds the user's post in the source discussion thread
THREAD_QUESTION = (
    "Is the “president for all Americans” idea still a meaningful standard, "
    "or has modern politics made that concept mostly obsolete?"
)
THREAD_URL = (
    "https://www.reddit.com/r/PoliticalDiscussion/comments/1tsemnh/"
    "is_the_idea_of_a_president_for_all_americans/"
)

FINETUNED = "Fine-tuned DistilBERT"
ZEROSHOT = "Zero-shot Groq (llama-3.3-70b)"

# lazily-loaded, cached so heavy imports / clients happen once
_model_cache = None
_groq_cache = None


def _load_finetuned():
    global _model_cache
    if _model_cache is None:
        if not os.path.isdir(MODEL_DIR):
            raise gr.Error(
                f"No fine-tuned model found at '{MODEL_DIR}'. "
                "Run `python train.py` first, or switch to the Zero-shot Groq model."
            )
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        _model_cache = (tokenizer, model, torch)
    return _model_cache


def _load_groq():
    global _groq_cache
    if _groq_cache is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise gr.Error("GROQ_API_KEY is not set. Add it to a .env file or your environment, "
                           "or switch to the Fine-tuned DistilBERT model.")
        try:
            from groq import Groq
        except ImportError:
            raise gr.Error("The 'groq' package is not installed. Run: pip install groq")
        _groq_cache = Groq()
    return _groq_cache


def _predict_finetuned(text):
    tokenizer, model, torch = _load_finetuned()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    scores = {ID_TO_LABEL[i]: float(probs[i]) for i in range(NUM_LABELS)}
    note = "Confidence scores are calibrated softmax probabilities over all four labels."
    return scores, note


def _predict_groq(text):
    client = _load_groq()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": classification_system_prompt()},
            {"role": "user", "content": classification_user_prompt(text)},
        ],
    )
    label = parse_label(resp.choices[0].message.content)
    if label is None:
        raise gr.Error(f"Could not parse a label from the model reply: "
                       f"{resp.choices[0].message.content!r}")
    # zero-shot returns a single choice — show it as 1.0 but label it as such
    scores = {label: 1.0}
    note = ("⚠️ Zero-shot returns a single hard choice, **not** a calibrated probability. "
            "The 100% reflects the model picking this label, not its certainty.")
    return scores, note


def predict(text, backend):
    if not text or not text.strip():
        raise gr.Error("Please write a post to classify.")
    if backend == ZEROSHOT:
        return _predict_groq(text)
    return _predict_finetuned(text)


# label glossary, sourced from labels.py
_glossary = "\n".join(f"- **{name}** — {definition}" for name, (_, definition) in LABEL_MAP.items())

with gr.Blocks(title="PolyAnalyzer") as demo:
    gr.Markdown(
        "# PolyAnalyzer\n"
        "Classify a political-discussion comment by **what kind of contribution it makes**. "
        "Pick a model, then write your own reply to the thread below and see the predicted "
        "label and confidence."
    )
    gr.Markdown(
        "### The discussion thread asks:\n"
        f"> *{THREAD_QUESTION}*\n\n"
        f"📎 **Read the full thread on Reddit:** [r/PoliticalDiscussion]({THREAD_URL}) — "
        "browse what others wrote and respond to a specific comment if you like.\n\n"
        "**Write your own response** — make an argument, give an opinion, react, or ask a "
        "question — and PolyAnalyzer will classify what kind of contribution it is.\n\n"
        "💡 **Want to stress-test the models?** Try writing a *purposefully misleading* post — "
        "e.g. dress up a bare opinion with fake-sounding statistics to bait an `analysis` label, "
        "or hide a real question inside an angry rant. See if you can fool the classifier, and "
        "compare how the two models react."
    )
    with gr.Row():
        with gr.Column(scale=3):
            backend = gr.Radio(
                choices=[ZEROSHOT, FINETUNED],
                value=ZEROSHOT,
                label="Model",
            )
            text_in = gr.Textbox(
                label="Your response to the thread",
                placeholder="Write your take on the question above...",
                lines=8,
            )
            with gr.Row():
                submit = gr.Button("Classify", variant="primary")
                clear = gr.ClearButton(text_in, value="Clear")
        with gr.Column(scale=2):
            label_out = gr.Label(num_top_classes=NUM_LABELS, label="Predicted label & confidence")
            note_out = gr.Markdown()
            gr.Markdown("### Labels\n" + _glossary)

    submit.click(predict, inputs=[text_in, backend], outputs=[label_out, note_out])
    text_in.submit(predict, inputs=[text_in, backend], outputs=[label_out, note_out])


if __name__ == "__main__":
    # share=True prints a temporary public *.gradio.live URL (valid ~72 hours) in
    # addition to the local one — handy for sharing a live demo with your project.
    demo.launch(share=True)
