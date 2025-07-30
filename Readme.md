

# ⚖️ LegalLabourAI – French Labour Code Assistant 🇫🇷

LegalLabourAI is a Streamlit application powered by large language models (LLMs) that provides accurate, context-aware answers based on the **French Labour Code**.

## 🔗 Useful Links

- 🚀 Live App: [LegalLabourAI on Streamlit](https://g6lbd5yieqgu5yvvrappuuv.streamlit.app/)
- 💻 Source Code: [GitHub Repository](https://github.com/sifaoufatai/LegalLabourAI)

---

## 🧠 Key Features

- 🔍 **Semantic search** across the French Labour Code.
- 🤖 **LLM selection**:
  - `Local model (Ollama/Mistral)`
  - `OpenAI GPT-3.5` (API key required)
- 🧾 **Conversation memory** to maintain context across user interactions.
- 🗣️ **Response adaptation** based on question type:
  - Greetings → friendly response in French.
  - Out-of-scope questions → polite redirect.
  - Labour law questions → answers strictly based on retrieved articles.
- 📚 **Cited sources** from the Labour Code in each answer.

---

## 🚀 Run Locally

### ✅ Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) (for using the local Mistral model)
- OpenAI API Key (if using GPT-3.5)

### 🛠️ Setup

```bash
git clone https://github.com/sifaoufatai/LegalLabourAI.git
cd LegalLabourAI
pip install -r requirements.txt
```

### ▶️ Start the App

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
LegalLabourAI/
├── app.py                         # Main Streamlit app
├── french_labour_code_vectordb/  # Chroma vector database (persistent embeddings)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🧰 Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Chroma DB](https://www.trychroma.com/)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- [Ollama + Mistral 7B](https://ollama.com/library/mistral)
- [OpenAI GPT-3.5](https://platform.openai.com/)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests to improve the app.

---

## 📜 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Fatai Idrissou**
AI Engineer & Master’s Student passionate about law and technology.
[GitHub Profile](https://github.com/sifaoufatai)
```

---

