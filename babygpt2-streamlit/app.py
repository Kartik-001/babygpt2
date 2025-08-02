# app.py
import streamlit as st
import requests

# Title
st.set_page_config(page_title="BabyGPT2 Chat", page_icon="🤖")
st.title("🤖 BabyGPT2 Chat")

# Instructions
st.markdown(
    """
Enter a prompt below and click **Generate**.  
Your text will be sent to the BabyGPT2 API and returned as generated continuation.
"""
)

# Text input box
prompt = st.text_area("Your prompt", height=150)

# Generate button
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        # Show spinner while waiting for response
        with st.spinner("Generating..."):
            try:
                # Call your hosted FastAPI endpoint
                resp = requests.post(
                    "https://babygpt2-api.onrender.com/generate",
                    headers={"Content-Type": "application/json"},
                    json={"prompt": prompt}
                )
                resp.raise_for_status()
                data = resp.json()
                generated = data.get("text", "")
            except Exception as e:
                st.error(f"API error: {e}")
                generated = ""

        # Display the result
        if generated:
            st.subheader("🖋️ Generated Text")
            st.write(generated)
