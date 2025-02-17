# **LangChain + Pinecone AI-PoweredRecommendation System**

Welcome to the **LangChain + Pinecone AI-Powered Recommendation System** project! This repository demonstrates how to build a recommendation system using **LangChain**, **OpenAI**, and **Pinecone**. The system allows users to input an item description and receive semantically similar recommendations based on vector embeddings.

![image](https://github.com/user-attachments/assets/01e6da0f-03c3-416a-8df6-f7e20bdac00a)


---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Setup Instructions](#setup-instructions)
5. [Backend Code](#backend-code)
6. [Frontend Code](#frontend-code)
7. [How It Works](#how-it-works)
8. [Use Cases](#use-cases)
9. [Contributing](#contributing)
10. [Contact Details](#contact-details)
11. [License](#license)

---

## **Project Overview**
This project combines **LangChain** (for language model integration) and **Pinecone** (for vector similarity search) to build a recommendation system. The backend is built using **FastAPI**, and the frontend is a simple **Streamlit** application. Users can input an item description, and the system will return semantically similar recommendations.

---

## **Features**
- **Semantic Search**: Retrieve recommendations based on semantic similarity.
- **Scalable**: Pinecone handles large-scale vector data efficiently.
- **User-Friendly Interface**: Streamlit provides an intuitive frontend for users.
- **Customizable**: Add or modify item descriptions to tailor the system to your needs.

---

## **Tech Stack**
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: OpenAI (via LangChain)
- **Environment Management**: `dotenv`

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/HaseebAhmed49/langchain-pinecone-recommendation-system.git
cd langchain-pinecone-recommendation-systems
```

### **2. Install Dependencies**
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3. Set Up Environment Variables**
Create a `.env` file in the root directory and add your API keys:
```plaintext
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

### **4. Run the Backend**
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
The backend will be available at `http://localhost:8000`.

### **5. Run the Frontend**
Start the Streamlit app:
```bash
streamlit run frontend.py
```
The frontend will be available at `http://localhost:8501`.

---

## **How Application Works**
1. **User Input**: The user enters an item description in the Streamlit frontend.
2. **Backend Processing**:
   - The backend generates an embedding for the input using OpenAI.
   - It queries the Pinecone index to find semantically similar items.
3. **Recommendations**: The backend returns the top-k recommendations, which are displayed in the frontend.

---

## **Use Cases**
- **E-commerce**: Recommend products based on user queries.
- **Content Platforms**: Suggest articles, videos, or podcasts.
- **Knowledge Management**: Retrieve relevant documents or FAQs.
- **Chatbots**: Enable context-aware conversations.

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## **Contact Details**
For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Name**: Haseeb Ahmed
- **Email**: haseebahmed02@gmail.com
- **LinkedIn:** [Haseeb Ahmed49](https://www.linkedin.com/in/haseebahmed49/)
- **GitHub:** [HaseebAhmed49](https://github.com/HaseebAhmed49)

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding! 🚀
