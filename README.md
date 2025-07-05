# 🤖 ScribeMate: NLP Chatbot for Lecture Notes & Question Generation

ScribeMate is an interactive chatbot built with Python and Streamlit that helps students process long lecture transcripts by automatically generating concise summaries and relevant study questions.

## ✨ Key Features

- **Conversational UI:** Interact with the bot in a natural, conversational flow.
- **Automated Summarization:** Utilizes the T5 transformer model to generate abstractive summaries of long texts.
- **Concept Extraction:** Employs a hybrid approach (NER, RAKE, TF) to identify key terms and concepts.
- **Question Generation:** Generates relevant study questions based on the transcript's content to aid revision.
- **Customizable Theme:** A clean, themed interface built with Streamlit.

## 🛠️ Technology Stack

- **Language:** Python 3.11
- **Core Libraries:** Streamlit, Transformers, PyTorch, spaCy, NLTK
- **Models:** T5-small (Summarization), potsawee/t5-large-generation-squad-QuestionAnswer (QG)

## ⚙️ Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[YOUR_REPO_NAME].git
    cd [YOUR_REPO_NAME]
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Use the Python version you developed with (e.g., 3.11)
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLP models:**
    ```bash
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('stopwords')"
    ```

## 🚀 How to Run

Ensure your virtual environment is activated. Run the following command in your terminal:

```bash
streamlit run app.py --server.fileWatcherType none
```
