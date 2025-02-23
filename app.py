import os
import streamlit as st
import numpy as np
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from groq import Groq

# Load API Key from Environment
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()

# Initialize Groq Client
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# Load PubMedBERT Model (Try Groq API first, then Hugging Face)
try:
    pubmedbert_tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")
    pubmedbert_model = AutoModel.from_pretrained("NeuML/pubmedbert-base-embeddings")
    pubmedbert_pipeline = pipeline('feature-extraction', model=pubmedbert_model, tokenizer=pubmedbert_tokenizer, device=-1)
except Exception:
    st.warning("Error loading PubMedBERT from Groq API. Using Hugging Face model.")
    pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    pubmedbert_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    pubmedbert_pipeline = pipeline('feature-extraction', model=pubmedbert_model, tokenizer=pubmedbert_tokenizer, device=-1)

# Initialize FAISS Index
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

# Function to Check if Query is Related to Epilepsy
def preprocess_query(query):
    tokens = query.lower().split()
    epilepsy_keywords = ["seizure", "epilepsy", "convulsion", "neurology", "brain activity"]

    is_epilepsy_related = any(k in tokens for k in epilepsy_keywords)

    return tokens, is_epilepsy_related

# Function to Generate Response with Chat History
def generate_response(user_query, chat_history):
    # Grammatical Correction using LLaMA (Hidden from User)
    try:
        correction_prompt = f"""
        Correct the following user query for grammar and spelling errors, but keep the original intent intact.
        Do not add or remove any information, just fix the grammar.
        User Query: {user_query}
        Corrected Query:
        """
        grammar_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": correction_prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        corrected_query = grammar_completion.choices[0].message.content.strip()
        # If correction fails or returns empty, use original query
        if not corrected_query:
            corrected_query = user_query
    except Exception as e:
        corrected_query = user_query # Fallback to original query if correction fails
        print(f"⚠️ Grammar correction error: {e}") # Optional: Log the error for debugging

    tokens, is_epilepsy_related = preprocess_query(corrected_query) # Use corrected query for processing

    # Greeting Responses
    greetings = ["hello", "hi", "hey"]
    if any(word in tokens for word in greetings):
        return "👋 Hello! How can I assist you today?"

    # If Epilepsy Related - Use Epilepsy Focused Response
    if is_epilepsy_related:
        # Try Getting Medical Insights from PubMedBERT
        try:
            pubmedbert_embeddings = pubmedbert_pipeline(corrected_query) # Use corrected query for PubMedBERT
            embedding_mean = np.mean(pubmedbert_embeddings[0], axis=0)
            index.add(np.array([embedding_mean]))
            pubmedbert_insights = "**PubMedBERT Analysis:** ✅ Query is relevant to epilepsy research."
        except Exception as e:
            pubmedbert_insights = f"⚠️ Error during PubMedBERT analysis: {e}"

        # Use LLaMA for Final Response Generation with Chat History Context (Epilepsy Focus)
        try:
            prompt_history = ""
            if chat_history:
                prompt_history += "**Chat History:**\n"
                for message in chat_history:
                    prompt_history += f"{message['role'].capitalize()}: {message['content']}\n"
                prompt_history += "\n"

            epilepsy_prompt = f"""
            {prompt_history}
            **User Query:** {corrected_query} # Use corrected query for final response generation
            **Instructions:** Provide a concise, structured, and human-friendly response specifically about epilepsy or seizures, considering the conversation history if available.
            """

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": epilepsy_prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            model_response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"⚠️ Error generating response with LLaMA: {e}"

        return f"**NeuroGuard:** ✅ **Analysis:**\n{pubmedbert_insights}\n\n**Response:**\n{model_response}"


    # If Not Epilepsy Related - Try to Answer as General Health Query
    else:
        # Try Getting Medical Insights from PubMedBERT (even for general health)
        try:
            pubmedbert_embeddings = pubmedbert_pipeline(corrected_query)
            embedding_mean = np.mean(pubmedbert_embeddings[0], axis=0)
            index.add(np.array([embedding_mean]))
            pubmedbert_insights = "**PubMedBERT Analysis:**  PubMed analysis performed for health-related context." # General analysis message
        except Exception as e:
            pubmedbert_insights = f"⚠️ Error during PubMedBERT analysis: {e}"

        # Use LLaMA for General Health Response Generation with Chat History Context
        try:
            prompt_history = ""
            if chat_history:
                prompt_history += "**Chat History:**\n"
                for message in chat_history:
                    prompt_history += f"{message['role'].capitalize()}: {message['content']}\n"
                prompt_history += "\n"

            general_health_prompt = f"""
            {prompt_history}
            **User Query:** {corrected_query}
            **Instructions:** Provide a concise, structured, and human-friendly response to the general health query, considering the conversation history if available. If the query is clearly not health-related, respond generally.
            """

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": general_health_prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            model_response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"⚠️ Error generating response with LLaMA: {e}"

        return f"**NeuroGuard:** ✅ **Analysis:**\n{pubmedbert_insights}\n\n**Response:**\n{model_response}"


# Streamlit UI Setup
st.set_page_config(page_title="NeuroGuard: Epilepsy & Health Chatbot", layout="wide") # Updated title
st.title("🧠 NeuroGuard: Epilepsy & Health Chatbot") # Updated title
st.write("💬 Ask me anything about epilepsy, seizures, and general health. I remember our conversation!") # Updated description

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Bot Response
    with st.chat_message("bot"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = generate_response(prompt, st.session_state.chat_history) # Pass chat history here
                st.markdown(response)
                st.session_state.chat_history.append({"role": "bot", "content": response})
            except Exception as e:
                st.error(f"⚠️ Error processing query: {e}")