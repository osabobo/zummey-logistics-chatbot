
# Zummey Logistics Chatbot

This project is a conversational AI chatbot for Zummey Logistics built with **Streamlit**, **LangChain**, **Groq LLM**, **FAISS**, and **SerpAPI**. It allows users to:

- Place delivery orders through a guided conversation
- Get information from a local knowledge base (vector DB)
- Search the web for Zummey Logistics-related questions (via SerpAPI)
- Save confirmed orders to Google Sheets

## 🚀 Features

1. **Order Placement**: The chatbot walks users through entering sender/receiver details and delivery instructions, validating inputs, and saving orders to Google Sheets.
2. **Knowledge Base Q&A**: Uses a FAISS vectorstore built from company docs (e.g., policies, FAQs) to answer internal queries.
3. **Live Web Search**: Leverages SerpAPI to answer queries not available in the internal DB.
4. **Streamlit UI**: Interactive chat interface with chat history, input validation, progress tracking, and more.

## 🛠 Setup Instructions

1. **Clone this repo**:

```bash
git clone https://github.com/your-username/zummey-logistics-chatbot.git
cd zummey-logistics-chatbot
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Environment Variables**

Create a `.env` file with the following:

```
GROQ_API_KEY=your_groq_key_here
SERPAPI_API_KEY=your_serpapi_key_here
CREDS_FILE_PATH=path_to_your_google_creds.json
SPREADSHEET_ID=your_google_sheets_id
```

4. **Prepare the Vector DB**

Ensure you've already embedded your documents into a FAISS index and saved it in:

```
vectorstoress/db_faiss
```

5. **Run the App**:

```bash
streamlit run cahat3.py
```

## 🧾 Order Fields Collected

- Sender Name
- Sender Phone
- Sender Email
- Receiver Name
- Receiver Phone
- pickup
- dropup
- Delivery Instructions

## 🧠 Technologies Used

- LangChain (ConversationalRetrievalChain, FAISS, LLMChain)
- HuggingFace Embeddings
- ChatGroq LLM
- Google Sheets API (via `gspread`)
- Streamlit
- SerpAPI (for live search)
- Python 3.10+

## 📄 License

MIT License
