# app.py
import streamlit as st
import requests

# Page config
st.set_page_config(page_title="BabyGPT2 Chat", page_icon="🤖")
st.title("🤖 BabyGPT2 Chat")
st.markdown("""
Enter a prompt below and either **press Enter** or click **Generate**.  
Your text will be sent to the BabyGPT2 API and its continuation will appear below.
""")

# Placeholder for output below the form
output_placeholder = st.empty()

# Define the generation function
def do_generate(prompt: str):
    with output_placeholder:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return
        with st.spinner("Generating..."):
            resp = requests.post(
                "https://babygpt2-api.onrender.com/generate",
                json={"prompt": prompt}
            )
            resp.raise_for_status()
            text = resp.json().get("text", "")
        st.subheader("🖋️ Generated Text")
        st.write(text)

# ▪️ FORM: wraps the input and submit button
with st.form(key="prompt_form"):
    user_input = st.text_input(
        "Your prompt (press Enter to send)", 
        key="prompt_input"
    )
    submit_btn = st.form_submit_button("Generate")

# After the form block, check if submitted
if submit_btn:
    do_generate(user_input)
