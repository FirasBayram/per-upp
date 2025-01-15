import streamlit as st
import pycountry
from openai import OpenAI

# Get list of countries (use pycountry library)
countries = [country.name for country in pycountry.countries]

def call_openai_api(prompt, api_key):
    """Send the prompt to OpenAI API using the fine-tuned model and return the response."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="ft:gpt-4-0824:karlstad-university:perupprefined:AIvn0RiI",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.title("Travel Planning App")
    
    # Step 1: Collect user information and travel preferences
    st.header("User Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    # Select country of residence from dropdown
    country_residence = st.selectbox("Country of Residence", countries)
    
    # City of residence input
    city_residence = st.text_input("City of Residence")
    
    # Situational Data
    st.header("Situational Data")
    days_for_visit = st.number_input("# of Days for Visit", min_value=1, value=1, step=1)
    transportation = st.selectbox("Type of Transportation", ["Own Car", "Public Transport"])
    
    # Travel Constellation
    st.subheader("Traveling Constellation")
    num_people = st.number_input("# of People", min_value=1, value=1, step=1)
    num_children = st.number_input("# of Children under 10?", min_value=0, value=0, step=1)
    point_of_departure = st.text_input("Point of Departure")
    
    # Area of Interest (Solution Space)
    st.header("Area of Interest")
    area_of_interest = st.selectbox("Choose an Area of Interest",
                                  ["Literature", "Science and Engineering", "Design and Art"])
    
    # Step 2: Enter OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    # Step 3: Generate travel plan
    if openai_api_key and st.button("Generate Travel Plan"):
        if not city_residence or not point_of_departure:
            st.error("Please fill in all required fields.")
            return
            
        normal_prompt = (
            f"Please suggest a travel plan based in Värmland based on the following details: "
            f"The user is {age} years old and identifies as {gender}. They reside in {city_residence}, "
            f"{country_residence} and plan to visit for {days_for_visit} days using {transportation}. "
            f"They will be traveling with {num_people} adults and {num_children} children under 10, "
            f"departing from {point_of_departure}. Their area of interest is {area_of_interest}."
        )
        
        # Display the generated prompt
        st.write("### Generated Prompt")
        st.write(normal_prompt)
        
        # Generate and display response
        with st.spinner('Generating your travel plan...'):
            response = call_openai_api(normal_prompt, openai_api_key)
            if response:
                st.write("### Your Travel Plan")
                st.write(response)

if __name__ == '__main__':
    main()
