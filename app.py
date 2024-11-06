import streamlit as st
import pycountry
import faiss
import numpy as np
import pickle
import requests
from openai import OpenAI
import os

# Get list of countries (use `pycountry` library)
countries = [country.name for country in pycountry.countries]

# Function to download and merge the split FAISS files
def download_and_merge_chunks(urls, output_file, chunk_prefix):
    with open(output_file, 'wb') as output:
        for i, url in enumerate(urls):
            # Download each chunk
            response = requests.get(url)
            with open(f"{chunk_prefix}_{i+1:03d}.bin", 'wb') as chunk_file:
                chunk_file.write(response.content)
            print(f"Downloaded: {chunk_prefix}_{i+1:03d}.bin")

        # After downloading all chunks, merge them
        merge_files(output_file, chunk_prefix, len(urls))

# Function to merge split files into one
def merge_files(output_file, chunk_prefix, num_chunks):
    with open(output_file, 'wb') as output:
        for i in range(num_chunks):
            chunk_name = f"{chunk_prefix}_{i+1:03d}.bin"
            with open(chunk_name, 'rb') as chunk_file:
                output.write(chunk_file.read())
            print(f"Merged: {chunk_name}")

# Function to load FAISS index and texts
def load_index_and_texts(index_file="faiss_index_batch-3-LARGE.bin", text_file="texts2.pkl"):
    index = faiss.read_index(index_file)
    with open(text_file, "rb") as f:
        texts = pickle.load(f)
    return index, texts

# Function to search the index for relevant contexts
def search_index(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return distances, indices

# Function to call OpenAI API with RAG
def call_openai_api_with_rag(prompt, api_key):
    client = OpenAI(api_key=api_key)
    try:
        # Load FAISS index and texts
        index, texts = load_index_and_texts()

        # Generate embedding for the prompt
        embedding_response = client.embeddings.create(input=[prompt], model="text-embedding-3-large")
        query_embedding = np.array(embedding_response.data[0].embedding)

        # Retrieve relevant contexts
        distances, indices = search_index(index, query_embedding)
        retrieved_texts = []
        similarity_threshold = 1.5  # Adjust for appropriate similarity

        # Collect top relevant texts
        for i in range(len(indices[0])):
            if distances[0][i] < similarity_threshold:
                idx = indices[0][i]
                if 0 <= idx < len(texts):
                    retrieved_texts.append(texts[idx])

        # Default message if no match found within threshold
        if not retrieved_texts:
            retrieved_texts.append("No specific events found. Here are general recommendations for Värmland.")

        # Combine retrieved texts for context
        combined_context = " ".join(retrieved_texts[:4])  # Limit to top 4 results

        # Prepare message for chat completion with context
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Context: {combined_context}"}
        ]

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app content
def main():
    st.title("Travel Planning App")

    # Step 1: Collect user information and travel preferences
    st.header("User Information")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Select country of residence from dropdown
    country_residence = st.selectbox("Country of Residence", countries)

    # City of residence input
    city_residence = st.text_input("City of Residence")

    # Situational Data
    st.header("Situational Data")
    days_for_visit = st.number_input("# of Days for Visit", min_value=1, step=1)
    transportation = st.selectbox("Type of Transportation", ["Own Car", "Public Transport"])

    # Travel Constellation
    st.subheader("Traveling Constellation")
    num_people = st.number_input("# of People", min_value=1, step=1)
    num_children = st.number_input("# of Children under 10?", min_value=0, step=1)
    point_of_departure = st.text_input("Point of Departure")

    # Area of Interest (Solution Space)
    st.header("Area of Interest")
    area_of_interest = st.selectbox("Choose an Area of Interest",
                                    ["Literature", "Science and Engineering", "Design and Art"])

    # Step 2: Enter OpenAI API key just before generating the plan
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Step 3: Generate travel plan only after the "Generate Travel Plan" button is pressed
    if openai_api_key:  # Only proceed if API key is entered
        if st.button("Generate Travel Plan"):
            normal_prompt = (
                f"Please suggest a travel plan based in Värmland based on the following details: "
                f"The user is {age} years old and identifies as {gender}. They reside in {city_residence}, "
                f"{country_residence} and plan to visit for {days_for_visit} days using {transportation}. "
                f"They will be traveling with {num_people} adults and {num_children} children under 10, "
                f"departing from {point_of_departure}. Their area of interest is {area_of_interest}."
            )

            # Display the generated prompt
            st.write("### Generated Prompt for LLM")
            st.write(normal_prompt)

            # Display spinner while waiting for response
            with st.spinner('Generating your travel plan...'):
                # Call OpenAI API with RAG and display the result
                response = call_openai_api_with_rag(normal_prompt, openai_api_key)

            # Display the result after spinner ends
            st.write("### Travel Plan Response from LLM")
            st.write(response)

# Run the app
if __name__ == '__main__':
    main()
