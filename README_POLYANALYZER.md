
# PolyAnalyzer

PolyAnalyzer is an interactive web app for classifying political discussion
comments by contribution type. It labels each response as one of four categories:
`analysis`, `op_response`, `response`, or `question`.

The app supports two model backends:
- **Zero-shot Groq (`llama-3.3-70b-versatile`)**: fast, API-based classification.
- **Fine-tuned DistilBERT (`distilbert-base-uncased`)**: local model inference from
	`./model` if you have already run training.

## What Users Should Expect

When users open PolyAnalyzer, they should expect to:
- See the seed thread question and a link to the original Reddit discussion.
- Select a model backend (Zero-shot Groq or Fine-tuned DistilBERT).
- Paste or write a comment in the input box and click **Classify**.
- Receive a predicted label and confidence-style output.

Output behavior depends on model selection:
- **Fine-tuned DistilBERT** returns calibrated softmax probabilities across all
	four labels.
- **Zero-shot Groq** returns a single hard label; the shown 100% indicates the
	selected class, not calibrated uncertainty.

If `./model` is missing and the user selects Fine-tuned DistilBERT, the app shows
an error prompting them to run `train.py`. If `GROQ_API_KEY` is missing and the user
selects Groq mode, the app prompts for API key setup.

## Run Instructions



### 1. Clone the repository

```bash
git clone https://github.com/mardisworld/AI201-project3-TakeMeter.git
cd AI201-project3-TakeMeter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

The app defaults to the Zero-shot Groq model, which needs a key (get one at
console.groq.com). Either set an environment variable:

```bash
export GROQ_API_KEY=your_key_here
```

Or create a `.env` file in the project folder (auto-loaded):

```env
GROQ_API_KEY=your_key_here
```

### 4. Launch the app

```bash
python app.py
```

Then open the URL it prints (default `http://127.0.0.1:7860`) in your browser.

That is all you need for the default Groq mode. Write a response to the thread
question, click Classify, and you will see the predicted label and confidence.

Optional: to enable the Fine-tuned DistilBERT toggle option, you first need a
trained model in `./model` (run train.py):

```bash
python preprocess.py --input ../Reddit-thread-on-American-presidency-v2.csv
python train.py

```

Without the /model, the Groq toggle still works; switching to the fine-tuned option
will show a message telling you to run `train.py` first.

