# --- Imports ---
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

# --- Setup ---
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a helpful logistics assistant for Zummey Logistics. Your task is to answer user enquiries and collect order details.
If the information asked by the user is not available, politely inform them and suggest alternative solutions.
At the end of the conversation, provide a summary of the collected information on order delivery.

Context: {context}
User: {question}
Chatbot:

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def retrieval_chatbot_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    load_dotenv() 
    KEY = os.getenv("GROQ_API_KEY")
    return ChatGroq(model="Llama-3-3-70b-Versatile", groq_api_key=KEY, temperature=0.8)

def chat_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) 
    return retrieval_chatbot_chain(load_llm(), set_custom_prompt(), db)

def final_result(query):
    chatbot_result = chat_bot()
    response = chatbot_result({'query': query})
    return response

# --- Google Sheets Setup ---
load_dotenv(dotenv_path=".env")
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = os.getenv("CREDS_FILE_PATH") # Get JSON key file path from environment variable
SPREADSHEET_ID = "" # Google Sheet ID
SHEET_NAME = "Order_Details" # Name of the work sheet

def save_order_to_sheets(order_details):
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

    # Add headers if sheet is empty
    if sheet.row_count == 0:
        headers = [key.replace('_', ' ').title() for key in order_details.keys()]
        sheet.append_row(headers)

    sheet.append_row(list(order_details.values()))
    return True

# --- Streamlit App Start ---
st.title("Zummey Logistics Chatbot")
st.subheader("Cheap and Easy Logistics")
st.write("Hi, Welcome to Zummey Logistics. I will be glad to assist with your order delivery.")

# --- Session State Init ---
for key, default in {
    "messages": [], "order_step": 0, "order_info": {},
    "order_submitted": False, "user_choice": "Place Order"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Chat Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Option ---
st.session_state.user_choice = st.sidebar.radio("What would you like to do?", ("Place Order", "Make Enquiry"))

# --- Place Order Flow ---
order_questions = [
    ("sender_name", "Hi, what's your name?"),
    ("sender_phone", "Thank you {sender_name}! What is your phone number?"),
    ("sender_email", "And your email address for further communication?"),
    ("receiver_name_intro", "{sender_name}, can you provide the receiver information?"),
    ("receiver_name", "What is the receiver's name?"),
    ("receiver_phone", "What is the receiver's phone number?"),
    ("instructions", "Any specific instructions for delivery?")
]

if st.session_state.user_choice == "Place Order":

    if st.session_state.order_submitted:
        st.success("Your order has been submitted successfully!")
        if st.button("Start New Order"):
            st.session_state.update({
                "order_step": 0,
                "order_info": {},
                "messages": [],
                "order_submitted": False
            })
        st.stop()

    step = st.session_state.order_step

    if step < len(order_questions):
        key, question = order_questions[step]
        try:
            question_text = question.format(**st.session_state.order_info)
        except:
            question_text = question

        with st.chat_message("chatbot"):
            st.markdown(question_text)

        user_input = st.chat_input("")

        if user_input:
            st.session_state.messages.append({"role": "chatbot", "content": question_text})
            st.session_state.messages.append({"role": "user", "content": user_input})

            if not key.endswith("_intro"):
                st.session_state.order_info[key] = user_input

            st.session_state.order_step += 1
            st.rerun()
    else:
        with st.chat_message("chatbot"):
            st.markdown("Thank you for providing your order details! Here's a summary for confirmation:")
            for key, value in st.session_state.order_info.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
            if st.button("Confirm & Submit"):
                if save_order_to_sheets(st.session_state.order_info):
                    st.session_state.order_submitted = True
                    st.rerun()

# --- Make Enquiry Flow ---
elif st.session_state.user_choice == "Make Enquiry":
    enquiry = st.chat_input("Enter your enquiry:")
    if enquiry:
        st.session_state.messages.append({"role": "user", "content": enquiry})
        with st.chat_message("user"):
            st.markdown(enquiry)

        response = final_result(enquiry)
        st.session_state.messages.append({"role": "chatbot", "content": response["result"]})
        with st.chat_message("chatbot"):
            st.markdown(response["result"])
