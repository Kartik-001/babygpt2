# app.py
import streamlit as st
import requests

# 1️⃣ Page config and title
st.set_page_config(page_title="BabyGPT2 Chat", page_icon="🤖")
st.title("🤖 BabyGPT2 Chat")

st.markdown(
    """
Enter a prompt below and either **press Enter** or click **Generate**.  
Your text will be sent to the BabyGPT2 API and the generated continuation will appear below.
"""
)

# 2️⃣ Define the generate-text callback
def generate():
    prompt = st.session_state.prompt  # get the current prompt
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        return

    with st.spinner("Generating..."):
        try:
            resp = requests.post(
                "https://babygpt2-api.onrender.com/generate",
                json={"prompt": prompt}
            )
            resp.raise_for_status()
            st.session_state.generated = resp.json().get("text", "")
        except Exception as e:
            st.session_state.generated = f"API error: {e}"

# Placeholder for spinner + results, placed exactly where YOU want them:
result_placeholder = st.container()

# Text input (with on_change callback)
st.text_input(
    "Your prompt (press Enter to send)",
    key="prompt",
    on_change=generate
)

# Generate button
if st.button("Generate"):
    generate()

# 5️⃣ Display the generated text (if any)
if "generated" in st.session_state:
    st.subheader("🖋️ Generated Text")
    st.write(st.session_state.generated)
