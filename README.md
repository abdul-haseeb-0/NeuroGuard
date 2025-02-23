# 🧠 NeuroGuard: Epilepsy & Health Chatbot

## 🚀 Overview
**NeuroGuard** is an AI-powered chatbot designed to provide insights on **epilepsy, seizures, and general health-related queries**. It integrates state-of-the-art **LLMs (LLaMA), PubMedBERT embeddings, FAISS**, and **Streamlit** for an interactive and intelligent healthcare assistant experience.

---

## 🔥 Features
✔ **Epilepsy & Seizure Insights** - Provides structured responses to epilepsy-related queries.  
✔ **Medical Text Processing** - Uses **PubMedBERT** for medical text analysis and embeddings.  
✔ **AI-Powered Chat Responses** - Generates responses using **LLaMA-3.3-70B-Versatile** for accurate health guidance.  
✔ **FAISS Indexing for Retrieval** - Stores medical embeddings for faster query retrieval.  
✔ **Grammatical Correction** - Automatically improves user queries before processing.  
✔ **Streamlit UI** - Interactive chat interface with chat history retention.  
✔ **Health & Wellness Tips** - Offers advice on general health topics like headaches, nutrition, and stress management.  
✔ **Dynamic Query Classification** - Distinguishes between epilepsy, healthcare, and general queries.  

---

## 🛠️ Tech Stack
🔹 **Python** - Core programming language  
🔹 **Streamlit** - UI framework for chatbot interaction  
🔹 **Transformers (Hugging Face)** - For LLM-based text processing  
🔹 **FAISS** - Efficient similarity search for embeddings  
🔹 **Groq API** - For LLaMA-powered responses  
🔹 **PubMedBERT** - Specialized model for medical-related queries  
🔹 **NumPy** - For numerical computations  

---

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/neuroguard-chatbot.git
cd neuroguard-chatbot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Add your **GROQ API Key** as an environment variable:
```bash
export GROQ_API_KEY="your_api_key_here"
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📌 Usage
1️⃣ Open the chatbot UI in your browser.  
2️⃣ Ask any health-related or epilepsy-specific question.  
3️⃣ Receive AI-generated insights, medical references, and recommendations.  
4️⃣ View your chat history for context-aware responses.  

---

## 🤖 How It Works
1️⃣ **User Input:** You enter a health-related query.  
2️⃣ **Grammar Correction:** LLaMA fixes grammar errors while maintaining intent.  
3️⃣ **Query Classification:** AI determines if the query is about **epilepsy, general health, or other topics**.  
4️⃣ **Response Generation:** 
   - If epilepsy-related, **PubMedBERT** embeddings are generated and stored in **FAISS**.  
   - If general health, basic **medical guidance** is provided.  
   - If unrelated, the chatbot suggests alternative resources.  
5️⃣ **Display Response:** The chatbot replies with structured insights.  

---

## 📝 Future Enhancements
✅ Expand medical coverage to more health conditions.  
✅ Improve response accuracy with fine-tuned models.  
✅ Integrate **speech-to-text** for voice-based interaction.  
✅ Enhance UI for better user experience.  

---

## 👥 Contributors
- **[Abdul Haseeb]** - Developer & AI Engineer  
- **[Amina Asif]** - Developer & AI Engineer  
- **Hugging Face** - Model Providers  
- **Groq API** - LLM Support  

---

## 📜 License
MIT License - Feel free to use, modify, and distribute!  

---

## ⭐ Support & Feedback
🔗 For feature requests or issues, open an **[issue](https://github.com/Abdul-Haseeb-AI/NeuroGuard/issues)**.  
💬 Connect with me on **[LinkedIn](https://www.linkedin.com/in/abdul-haseeb-980075323/)**.  
🚀 If you like this project, consider giving it a **⭐ Star** on GitHub!