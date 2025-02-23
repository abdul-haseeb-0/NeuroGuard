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
    # Expanded keyword list for epilepsy and seizures
    epilepsy_keywords = ["seizure", "seizures", "epilepsy", "convulsion", "convulsions",
                         "symptoms", "jerks", "confusion", "aura", "tonic-clonic",
                         "absence", "myoclonic", "atonic", "focal", "generalized",
                         "treatment", "medication", "antiepileptic", "AEDs", "therapy",
                         "diagnosis", "triggers", "first aid", "emergency", "rescue",
                         "vns", "ketogenic diet", "surgery", "neurologist", "EEG", "MRI",
                         "ictal", "postictal", "interictal", "status epilepticus",
                         "epileptic", "brain", "neurological"]

    keywords = [keyword for keyword in tokens if keyword in epilepsy_keywords]
    is_epilepsy_related = any(keyword in keywords for keyword in epilepsy_keywords) # More specific check

    return tokens, keywords, is_epilepsy_related

def generate_response(user_query):
    tokens, keywords, is_epilepsy_related = preprocess_query(user_query)
    enhanced_query = " ".join(tokens)
    symptom_insights = ""
    condition_detected = "None" # Initialize condition detection
    recommendations = "" # Initialize recommendations
    greeting_response = "" # Initialize greeting response

    conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.current_conversation_messages])

    # User-friendly responses for greetings and specific queries
    if user_query.lower() in ["hi", "hello", "hey"]:
        greeting_response = "Hello there! How can I help you with epilepsy and seizure related questions today?"
        model_response = greeting_response # For direct greeting response
        final_response = f"Chatbot Response:\n{model_response}" # Simpler final response for greetings
        return final_response, model_response

    if user_query.lower() == "i have a headache":
        greeting_response = "I understand you have a headache. While headaches themselves are not always related to epilepsy or seizures, they can sometimes be a symptom or trigger for some people.  If your headache is new, severe, or accompanied by other symptoms like visual disturbances, confusion, jerking movements, or loss of consciousness, it's important to seek medical advice to rule out any underlying conditions.  Do you have any concerns about seizures or epilepsy in relation to your headache?"
        model_response = greeting_response
        final_response = f"Chatbot Response:\n{model_response}" # Simpler final response for headache query
        return final_response, model_response


    if is_epilepsy_related:
        try:
            pubmedbert_embeddings = pubmedbert_pipeline(user_query)
            embedding_mean = np.mean(pubmedbert_embeddings[0], axis=0)
            st.session_state.embeddings.append(embedding_mean)
            index.add(np.array([embedding_mean]))
            pubmedbert_insights = "PubMedBERT analysis of symptoms..."
            model_name = "PubMedBERT (Symptom Analysis)"
            model_response = pubmedbert_insights

            # Symptom and Condition Detection (Basic Keyword-based)
            symptom_keywords = ["jerks", "confusion", "staring", "loss of awareness", "falling", "stiffening", "twitching", "uncontrolled movements", "changes in sensation", "odd smells", "tastes", "feelings", "déjà vu", "jamais vu", "headache"] # Added headache to symptom keywords
            condition_keywords = {
                "tonic-clonic seizure": ["tonic-clonic", "grand mal", "stiffening and jerking"],
                "absence seizure": ["absence", "petit mal", "staring spells", "brief loss of awareness"],
                "focal seizure": ["focal", "partial", "localized", "one side of brain"],
                "myoclonic seizure": ["myoclonic", "muscle jerks", "sudden jerks", "brief jerks"],
                "atonic seizure": ["atonic", "drop attacks", "loss of muscle tone", "sudden falls"],
                "status epilepticus": ["status epilepticus", "prolonged seizure", "seizure lasting"],
                "headache related to seizure": ["headache", "ictal headache", "postictal headache", "head pain with seizure"] # Example condition with headache
            }

            detected_symptoms = [keyword for keyword in tokens if keyword in symptom_keywords]
            if detected_symptoms:
                symptom_insights = f"\n\n**Symptoms Detected:** {', '.join(detected_symptoms)}"

            for condition, condition_keys in condition_keywords.items():
                if any(key in keywords for key in condition_keys):
                    condition_detected = condition
                    break # Stop after first condition is detected for simplicity

            if condition_detected != "None":
                condition_insights = f"\n\n**Possible Condition (Keyword Indication):** {condition_detected}. Please note this is a preliminary keyword-based indication and not a diagnosis."
            else:
                condition_insights = ""

            if detected_symptoms: # Provide recommendations if symptoms are detected
                recommendations = """
                \n\n**General Recommendations (If Experiencing Symptoms):**

                * **Seek Immediate Medical Attention:** If you are experiencing new or worsening symptoms, it is crucial to consult a doctor or healthcare professional as soon as possible for proper evaluation and diagnosis.
                * **Do Not Drive or Operate Heavy Machinery:** If you suspect you might have seizures, avoid activities that could be dangerous if you were to have a seizure unexpectedly.
                * **Note Down Symptoms:** Keep a record of your symptoms, when they occur, how long they last, and any potential triggers. This information can be valuable for medical professionals.
                * **Ensure Safety:** If you have had a seizure before or are prone to seizures, take precautions to ensure your safety during a seizure. This might include informing people around you, avoiding hazardous situations, and making your environment safer.
                * **Follow Medical Advice:** If you have already been diagnosed with epilepsy or a seizure disorder, adhere strictly to your prescribed treatment plan and recommendations from your healthcare provider.
                """ # General, safe recommendations. **CRUCIAL: No specific medical advice.**
            else:
                recommendations = ""


        except Exception as e:
            model_response = f"Error during PubMedBERT analysis: {e}"
            recommendations = ""
            condition_insights = ""
            symptom_insights = ""


    else: # Non-epilepsy related query handling remains the same
        model_name = "LLaMA 2 / Mistral 7B (via Groq)"
        try:
            prompt =  """
            Conversation History:
            {conversation_history}

            User: {user_query}
            Bot: As an Epilepsy and Seizure Chatbot, I am specialized in providing information and support related to epilepsy and seizures.

            While I can process general questions, my expertise is focused on these specific neurological conditions.

            For general inquiries outside of epilepsy and seizures, you might find broader language models more helpful.

            For questions directly related to epilepsy and seizures, please proceed, and I will do my best to assist you with accurate and relevant information.

            Given this focus, and considering the user's last query, please provide a helpful and relevant response if the query is related to epilepsy or seizures. If the query is clearly outside this scope, gently guide the user back to the topic or indicate that the question is outside my area of expertise.

            User Query: {user_query}
            Response:
            """
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            model_response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"Error from Groq: {e}"
        recommendations = ""
        condition_insights = ""
        symptom_insights = ""


    final_response = f"**Enhanced Query:** {enhanced_query}\n\nChatbot Analysis (PubMedBERT if symptoms detected, else LLaMA 2):...\n{symptom_insights}{condition_insights}\n\nModel Response/Insights:\n{model_response}\n{recommendations}"
    if greeting_response: # Prioritize greeting response if it's set
        final_response = greeting_response
    return final_response, model_response

# Streamlit Interface (and other parts)
st.set_page_config(page_title="Epilepsy Chatbot", layout="wide") # Updated page title
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

st.title("Epilepsy & Seizure Chatbot") # More specific title
st.markdown(" **Disclaimer:** This Epilepsy Chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing symptoms of epilepsy or seizures, or have any concerns about your health, please consult a qualified healthcare professional immediately.  Do not use this chatbot to make decisions about your health or treatment.  Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.") # Disclaimer added
st.write("Welcome! This chatbot is designed to provide information and support related to epilepsy and seizures. Ask your questions about symptoms, treatments, diagnosis, and more.") # Clear introductory text


for message in st.session_state.current_conversation_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your epilepsy or seizure related query here:"): # Updated chat input placeholder
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