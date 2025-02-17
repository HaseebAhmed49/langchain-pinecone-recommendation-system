import streamlit as st
import requests

# Streamlit app title
st.title("Recommendation System with LangChain, OpenAI, and Pinecone")

# Input for item description
item_description = st.text_input("Enter an item description:")
k = st.slider("Number of recommendations to retrieve", min_value=1, max_value=10, value=5)

# Button to trigger recommendations
if st.button("Get Recommendations"):
    if item_description:
        # Send request to FastAPI backend
        response = requests.post(
            "http://localhost:8000/recommend",
            json={"item_description": item_description, "k": k}
        )

        if response.status_code == 200:
            recommendations = response.json().get("recommendations", [])
            if recommendations:
                st.write("### Recommendations:")
                for i, rec in enumerate(recommendations):
                    st.write(f"{i + 1}. {rec['item']}")
            else:
                st.write("No recommendations found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter an item description.")