import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from groq import Groq
import io
import faiss

# Groq API Key (and other initializations)
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

try:
    pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    pubmedbert_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    pubmedbert_pipeline = pipeline('feature-extraction', model=pubmedbert_model, tokenizer=pubmedbert_tokenizer, device=-1)
except Exception as e:
    st.error(f"Error loading PubMedBERT: {e}")
    st.stop()

embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

if "all_conversations" not in st.session_state:
    st.session_state.all_conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = 0
if "current_conversation_messages" not in st.session_state:
    st.session_state.current_conversation_messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# Functions
def preprocess_query(query):
    tokens = query.lower().split()
    keywords = [keyword for keyword in tokens if keyword in ["seizure", "symptoms", "jerks", "confusion", "epilepsy"]]
    is_medical_related = any(keyword in keywords for keyword in ["seizure", "symptoms", "jerks", "confusion", "epilepsy", "medical"])
    return tokens, keywords, is_medical_related

def generate_response(user_query):
    tokens, keywords, is_medical_related = preprocess_query(user_query)
    enhanced_query = " ".join(tokens)
    symptom_insights = ""

    conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.current_conversation_messages])

    if is_medical_related:
        try:
            pubmedbert_embeddings = pubmedbert_pipeline(user_query)
            embedding_mean = np.mean(pubmedbert_embeddings[0], axis=0)
            st.session_state.embeddings.append(embedding_mean)
            index.add(np.array([embedding_mean]))
            pubmedbert_insights = "PubMedBERT analysis..."
            model_name = "PubMedBERT"
            model_response = pubmedbert_insights
            if "seizure" in keywords or "symptoms" in keywords:
                remedy_recommendations = "\n\n**General Recommendations:**\n..."
            else:
                remedy_recommendations = ""
        except Exception as e:
            model_response = f"Error during PubMedBERT: {e}"
            remedy_recommendations = ""
    else:
        model_name = "LLaMA 2 / Mistral 7B (via Groq)"
        try:
            prompt = f"""
            Conversation History:
            {conversation_history}

            User: {user_query}
            Bot:
            """
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            model_response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"Error from Groq: {e}"
        remedy_recommendations = ""

    final_response = f"**Enhanced Query:** {enhanced_query}\n\nChatbot Analysis:...\n\nModel Response/Insights:\n{model_response}\n{remedy_recommendations}"
    return final_response, model_response
# Streamlit Interface (and other parts)
st.set_page_config(page_title="Epilepsy Chatbot", layout="wide")
st.markdown("<style>.chat-message.user {background-color: #e6f7ff; padding: 8px; border-radius: 8px; margin-bottom: 8px;}.chat-message.bot {background-color: #f0f0f0; padding: 8px; border-radius: 8px; margin-bottom: 8px;}.stTextArea textarea {background-color: #f8f8f8;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Conversations")
    if st.button("New Conversation"):
        st.session_state.current_conversation_id += 1
        st.session_state.current_conversation_messages = []
        st.session_state.embeddings = []
        index.reset()
    for conv_id in st.session_state.all_conversations:
        if st.button(f"Conversation {conv_id}"):
            st.session_state.current_conversation_id = conv_id
            st.session_state.current_conversation_messages = st.session_state.all_conversations[conv_id]
            st.session_state.embeddings = []
            index.reset()

st.title("Epilepsy & Seizure Chatbot")
st.write("Ask questions related to epilepsy and seizures.")

for message in st.session_state.current_conversation_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your query here:"):
    st.session_state.current_conversation_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("bot"):
        with st.spinner("Generating response..."):
            try:
                full_response, model_only_response = generate_response(prompt)
                st.markdown(model_only_response)
                st.session_state.current_conversation_messages.append({"role": "bot", "content": model_only_response})
            except Exception as e:
                st.error(f"Error processing query: {e}")

    st.session_state.all_conversations[st.session_state.current_conversation_id] = st.session_state.current_conversation_messages

# Download Chat
if st.session_state.current_conversation_messages:
    conversation_text = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in st.session_state.current_conversation_messages])
    st.download_button("Download Chat", data=conversation_text, file_name=f"chat_history_{st.session_state.current_conversation_id}.txt")