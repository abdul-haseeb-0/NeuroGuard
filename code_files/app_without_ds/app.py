import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from groq import Groq

groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set.")

client = Groq(api_key=groq_api_key)

# Load PubMedBERT for medical queries and embeddings
pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
pubmedbert_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
pubmedbert_pipeline = pipeline('feature-extraction', model=pubmedbert_model, tokenizer=pubmedbert_tokenizer, device=-1) # Use device=-1 for CPU

# Function to preprocess and understand user query
def preprocess_query(query):
    tokens = query.lower().split()
    keywords = [keyword for keyword in tokens if keyword in ["seizure", "symptoms", "jerks", "confusion", "epilepsy"]] # Example keywords
    is_medical_related = any(keyword in keywords for keyword in ["seizure", "symptoms", "jerks", "confusion", "epilepsy", "medical"]) # More comprehensive medical keyword check
    return tokens, keywords, is_medical_related

# Function to generate response (modified for structured output)
def generate_response(user_query):
    """
    Generates a structured response based on the user query, using PubMedBERT for medical analysis when relevant,
    and Groq for general conversation. Datasets and FAISS are removed in this version.
    """
    tokens, keywords, is_medical_related = preprocess_query(user_query)

    # Enhanced Query (for demonstration - can be improved further)
    enhanced_query = " ".join(tokens) # Basic enhancement - can add synonym expansion etc.

    symptom_insights = "" # FAISS retrieval removed

    if is_medical_related:
        # Use PubMedBERT for medical-related queries
        pubmedbert_embeddings = pubmedbert_pipeline(user_query)
        embedding_mean = np.mean(pubmedbert_embeddings[0], axis=0) # Calculate mean embedding

        pubmedbert_insights = "PubMedBERT analysis: (Embedding vector calculated for medical insight. Note: This version uses feature extraction for analysis and keyword-based recommendations, not direct seizure prediction. For specific predictions or detailed remedies, further fine-tuning and specialized models would be required.)"

        model_name = "PubMedBERT"
        model_response = pubmedbert_insights

        if "seizure" in keywords or "symptoms" in keywords:
            remedy_recommendations = "\n\n**General Recommendations & Remedies (Keyword-Based):**\n"
            remedy_recommendations += "- **Important:** Consult a neurologist for accurate diagnosis and personalized treatment.\n"
            remedy_recommendations += "- Describe your symptoms in detail to your doctor, including frequency, triggers, and duration.\n"
            remedy_recommendations += "- Maintain a seizure diary to track events and potential triggers.\n"
            remedy_recommendations += "- Ensure adequate sleep and stress management as potential seizure triggers.\n"
            remedy_recommendations += "- Follow your neurologist's advice regarding medication and lifestyle adjustments.\n"
        else:
            remedy_recommendations = ""
    else:
        # Use LLaMA 2 or Mistral 7B for general conversation (via Groq API)
        model_name = "LLaMA 2 / Mistral 7B (via Groq)"
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_query}],
                model="llama-3.3-70b-versatile", # Or try "mistral-8x7b-32768"
                stream=False,
            )
            model_response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"Error from Groq model: {e}"
        remedy_recommendations = ""


    # Merging insights and generating final structured response
    final_response = f"**Enhanced Query:** {enhanced_query}\n\n" # Section 1: Enhanced Query
    final_response += f"**Chatbot Analysis:**\n"
    final_response += f"- Query Type: {'Medical-related' if is_medical_related else 'General Conversation'}\n"
    final_response += f"- Model Used: {model_name}\n"
    final_response += f"- Keywords detected: {', '.join(keywords) if keywords else 'None'}\n" # Keywords section

    final_response += symptom_insights # Symptom insights removed from response
    final_response += f"\n**Model Response/Insights:**\n{model_response}\n" # Model-specific insights
    final_response += remedy_recommendations # Section 3: Remedy/Recommendations

    return final_response

# Streamlit Interface
st.title("Epilepsy & Seizure Chatbot (PubMedBERT & Groq)")
st.write("Ask questions related to epilepsy and seizures. This chatbot uses PubMedBERT for medical analysis and Groq (LLaMA 2/Mistral) for general conversation. Datasets and FAISS similarity search have been removed.")

user_query = st.text_area("Enter your query here:", placeholder="I'm experiencing sudden muscle jerks and confusion. What could this mean?")

if st.button("Get Response"):
    if user_query:
        response = generate_response(user_query)
        st.markdown(response) # Use st.markdown to render formatted response
    else:
        st.warning("Please enter a query.")