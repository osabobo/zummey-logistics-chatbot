# --- Imports ---
import os, re
import streamlit as st
import gspread
from pathlib import Path
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_groq import ChatGroq

# --- Load Environment Variables ---
load_dotenv()

DB_FAISS_PATH = "vectorstoress/db_faiss"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = os.getenv("CREDS_FILE_PATH")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SHEET_NAME = "Order_Details"

# --- Prompt Template ---
def get_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful logistics assistant for Zummey Logistics. 
Answer user enquiries and collect order details. If information isn't available, politely inform them.

Context: {context}
User: {question}
Chatbot:"""
    )

# --- Load LLM, Embeddings, Vector Store ---
def load_llm():
    #groq_key = os.getenv("GROQ_API_KEY")
    key = st.secrets["GROQ_API_KEY"]
    if not groq_key:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.8)

custom_prompt = get_custom_prompt()
llm = load_llm()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

def get_answer(query):
    response = qa_chain({"question": query})
    return {
        "result": response["answer"],
        "source_documents": response.get("source_documents", [])
    }

# --- Input Validation ---
def validate_input(field, value):
    value = str(value).strip()
    if not value:
        return "This field cannot be empty."

    if "name" in field:
        if not (8 <= len(value) <= 100):
            return "Name should be 8-100 characters long."
        if not re.match(r"^[a-zA-Z\s\-\.\']+$", value):
            return "Name can only contain letters, spaces, hyphens, periods, and apostrophes."

    elif "phone" in field:
        if not re.match(r"^\+?[\d\s\-()]{7,20}$", value) or sum(c.isdigit() for c in value) < 7:
            return "Enter a valid phone number with at least 7 digits."

    elif "email" in field:
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value):
            return "Enter a valid email address."
        if len(value) > 100:
            return "Email must be under 100 characters."

    elif "instructions" in field:
        if len(value) > 200:
            return "Instructions must be under 200 characters."

    return "Valid"

# --- Google Sheets Save ---
def save_order(order_details):
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        if sheet.row_count == 0:
            sheet.append_row([key.replace("_", " ").title() for key in order_details.keys()])
        sheet.append_row(list(order_details.values()))
        return True
    except Exception as e:
        st.error(f"❌ Failed to save to Google Sheets: {e}")
        return False

# --- Web Search Agent ---
def create_web_agent():
    #serp_api_key = os.getenv("SERPAPI_API_KEY")
    serp_api_key = st.secrets["SERPAPI_API_KEY"]
    if not serp_api_key:
        raise ValueError("Missing SERPAPI_API_KEY in .env")

    search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    tools = [Tool(name="Zummey Web Search", func=search.run, description="Live web search for Zummey Logistics")]
    return initialize_agent(tools, llm=load_llm(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Zummey Logistics Chatbot", layout="centered")
st.title("📦 Zummey Logistics Chatbot")
st.subheader("Cheap and Easy Delivery Services")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_choice" not in st.session_state:
    st.session_state.user_choice = "Make Enquiry"

# --- Sidebar ---
choice = st.sidebar.radio("Choose an action:", ("Place Order", "Make Enquiry"))
if choice != st.session_state.user_choice:
    st.session_state.user_choice = choice
    st.session_state.messages = []
    st.session_state.order_info = {}
    st.session_state.order_step = 0
    st.session_state.last_question = None
    st.session_state.last_input = None
    st.session_state.validation_error = None
    st.rerun()

# --- Chat History Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Order Placement Flow ---
if st.session_state.user_choice == "Place Order":
    fields = ["sender_name", "sender_phone", "sender_email",
              "receiver_name", "receiver_phone", "pickup", "dropup", "delivery_instructions"]
    current_step = len(st.session_state.order_info)
    st.progress(current_step / len(fields), f"Order Progress: {current_step}/{len(fields)}")

    if current_step < len(fields):
        current_field = fields[current_step]

        if st.session_state.get("validation_error"):
            prompt = f"""You are a friendly logistics assistant. The user entered '{st.session_state.last_input}' 
for '{current_field.replace('_', ' ')}', but it was invalid because: {st.session_state.validation_error}.
Kindly ask again and offer helpful input tips."""
        else:
            prompt = f"""You are a friendly logistics assistant. Given the data so far:
{st.session_state.order_info}
Ask the user to provide '{current_field.replace('_', ' ')}'. Provide a clear example like: 'e.g., John Doe'."""

        #llm_response = get_answer(prompt)["result"]
        if st.session_state.last_question != prompt:
            llm_response = get_answer(prompt)["result"]
            st.chat_message("chatbot").markdown(llm_response)
            st.session_state.messages.append({"role": "chatbot", "content": llm_response})
            st.session_state.last_question = prompt

        if user_input := st.chat_input("Answer or ask a question..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            if re.match(r"\b(what|how|when|can|do|is|are|should|could|why|where|who)\b", user_input.lower()):
                bot_reply = get_answer(user_input)["result"]
                st.chat_message("chatbot").markdown(bot_reply)
                st.session_state.messages.append({"role": "chatbot", "content": bot_reply})
                st.stop()

            validation = validate_input(current_field, user_input)
            if validation != "Valid":
                st.session_state.validation_error = validation
                st.session_state.last_input = user_input
                st.rerun()

            st.session_state.order_info[current_field] = user_input
            st.session_state.validation_error = None
            st.session_state.last_question = None
            st.rerun()
    else:
        st.chat_message("chatbot").markdown("### ✅ Here's your order summary:")
        for k, v in st.session_state.order_info.items():
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

        if st.button("Confirm & Submit Order"):
            if save_order(st.session_state.order_info):
                st.success("🎉 Order submitted successfully!")
                st.session_state.order_info = {}
                st.session_state.messages = []
                st.session_state.last_question = None
                st.rerun()

# --- Enquiry Flow ---
elif st.session_state.user_choice == "Make Enquiry":
    st.info("Ask anything about Zummey Logistics. We'll check our knowledge base, and search the web if needed.")
    if enquiry := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(enquiry)
        st.session_state.messages.append({"role": "user", "content": enquiry})

        response = get_answer(enquiry)
        reply = response["result"]

        if reply.strip().lower() in ["i don't know.", "i'm not sure.", "no relevant information found."]:
            with st.spinner("No info found locally. Searching the web..."):
                try:
                    agent = create_web_agent()
                    reply = agent.run(enquiry)
                except Exception as e:
                    reply = f"❌ Web search failed: {e}"

        st.chat_message("chatbot").markdown(reply)
        st.session_state.messages.append({"role": "chatbot", "content": reply})

