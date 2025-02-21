# NeuroGuard
Your intelligent assistant for predicting and managing epilepsy and seizures with personalized recommendations.

---

## 📌 **Working of the Epilepsy & Seizure Prediction Chatbot**

#### 1️⃣ **User Input**  
- The chatbot receives a user query such as:  
  > "I'm experiencing sudden muscle jerks and confusion. What could this mean?"

#### 2️⃣ **Preprocessing & Understanding the Query**  
- **Text Tokenization**: The user input is converted into tokens for processing.  
- **Keyword Detection**: Key symptoms and terms (e.g., seizure, muscle jerks, confusion) are identified.  
- **Query Classification**: The system determines if the query is **medical** or **general**.

#### 3️⃣ **Retrieval from FAISS Database**  
- The chatbot searches for similar past cases and **medical records** stored in the **FAISS database**.  
- Historical seizure cases or previous responses are retrieved to match the user query.

#### 4️⃣ **Choosing the Right Model for Response**  
- **Medical Query**: If the query is medical-related, **PubMedBERT** is used for accurate and precise medical advice.  
- **General Query**: If the query is a general conversation, models like **LLaMA 2** or **Mistral 7B** are used for a friendly and casual response.  
- **Prediction**: For seizure prediction, insights from both **FAISS** and **PubMedBERT** are combined.

#### 5️⃣ **Generating the Response**  
- **PubMedBERT** analyzes medical aspects such as symptoms, risks, and historical data.  
- **LLaMA 2 / Mistral** refines the response to ensure clarity, empathy, and a conversational tone.  
- The chatbot merges the insights from both models to form an accurate and useful reply.

#### 6️⃣ **Providing Recommendations & Predictions**  
If the input matches seizure symptoms, the chatbot provides:
- ✅ **Seizure Type Predictions** based on the symptoms.  
- ✅ **Risk Assessment** using historical data and medical knowledge.  
- ✅ **Medical Recommendations**: E.g., "Consult a neurologist", "Avoid known triggers".

#### 7️⃣ **Final Output to User**  
- The chatbot formats the final response in a **clear and user-friendly** manner.  
- Follow-up questions may be asked if needed for better accuracy.

#### 8️⃣ **Storing User Interaction (Optional)**  
- New user data and interactions can be stored in the **FAISS** database for future reference.  
- This helps the chatbot **improve over time** by learning from more case-based interactions.

---

## 🚀 **Key Features**  
- **Real-time Predictions**: Accurate seizure predictions based on symptoms.  
- **Medical Recommendations**: Informed advice using PubMedBERT.  
- **User Data Storage**: Improve model predictions with every interaction.  
- **Multi-model Approach**: Combining the power of **PubMedBERT**, **LLaMA 2**, and **FAISS**.

---

## 🛠️ **Tech Stack**  
- **Models**: PubMedBERT, LLaMA 2, Mistral 7B  
- **Database**: FAISS  
- **Programming**: Python
- **Deployment**: Hugging Face, Google Colab

---

**Hugging Face Link :** ```https://huggingface.co/spaces/Haseeb-001/NeuroGuard/tree/main``` 

---

## 💡 **How to Use**  
1. Clone this repository ``git clone https://github.com/Abdul-Haseeb-AI/NeuroGuard.git``
2. Install required dependencies  
3. Start the chatbot using **FastAPI**  
4. Interact with the chatbot and receive medical recommendations

---

## 💬 **Contribute**  
We welcome contributions to enhance the chatbot's features and improve its accuracy. Feel free to fork the repository and submit your pull requests! 🚀

---

📌 **If you find this project helpful, please ⭐ star the repo!** Let's improve seizure prediction together! 🚑🤖  
