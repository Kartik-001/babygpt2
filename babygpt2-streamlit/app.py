# app.py
import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="BabyGPT2 Chat", page_icon="🤖")
st.title("🤖 BabyGPT2 Chat")
st.markdown("""
Enter a prompt below and either **press Enter** or click **Generate**.  
Your text will be sent to the BabyGPT2 API and its continuation will appear below.
""")

# Input and button first
st.text_input(
    "Your prompt (press Enter to send)",
    key="prompt",
    on_change=lambda: generate()  # note: we'll define generate() below
)
generate_clicked = st.button("Generate")

# ➡️ Now create the placeholder *after* the button
result_placeholder = st.empty()

# Define the generate() function here so on_change can call it
def generate():
    prompt = st.session_state.prompt.strip()
    if not prompt:
        result_placeholder.warning("Please enter a prompt first.")
        return

    with result_placeholder:
        with st.spinner("Generating..."):
            resp = requests.post(
                "https://babygpt2-api.onrender.com/generate",
                json={"prompt": prompt}
            )
            resp.raise_for_status()
            text = resp.json().get("text", "")
        st.subheader("🖋️ Generated Text")
        st.write(text)

# If the button was clicked, run generate() too
if generate_clicked:
    generate()
