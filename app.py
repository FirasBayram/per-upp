import streamlit as st
import pycountry
from openai import OpenAI  # Import OpenAI client

# Get list of countries (use pycountry library)
countries = [country.name for country in pycountry.countries]


def call_openai_api(prompt, api_key):
    """Send the prompt to OpenAI API using the fine-tuned model and return the response."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:karlstad-university:perupprefined:AIvn0RiI",
            messages=[{"role": "user", "content": prompt}]
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
                # Send the normal prompt to OpenAI's API and display the result
                response = call_openai_api(normal_prompt, openai_api_key)

            # Display the result after spinner ends
            st.write("### Travel Plan Response from LLM")
            st.write(response)


# Run the app
if __name__ == '__main__':
    main()
