from openai import OpenAI
import streamlit as st

f = open(r"C:\Users\seela\OneDrive\Desktop\GenAI\GenAI_APP\API_Key.txt")

OPENAI_API_KEY = f.read()

client = OpenAI(api_key = OPENAI_API_KEY)

st.title("CODE DEBUGGER")

prompt=st.text_area("Enter your code:  ",height=150)
if st.button("Generate 🙂") == True:
    response = client.chat.completions.create(
               model="gpt-3.5-turbo-0301",
               messages=[
               {"role": "system", "content": "You are a code debugging assistent.Your task is to identify errors and fix the errors in the prompt."},
               
               {"role": "user", "content": prompt}
               ]
        )
    st.write("Output: ")
    st.write(response.choices[0].message.content)