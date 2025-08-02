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

# 1) Prompt input (single-line, captures Enter)
st.text_input(
    "Your prompt", 
    key="prompt", 
    on_change=lambda: st.session_state.generate_clicks.append("enter")
)

# 2) Generate button (captures clicks)
if st.button("Generate"):
    st.session_state.generate_clicks.append("button")

# Initialize a list in session_state to track submissions
if "generate_clicks" not in st.session_state:
    st.session_state.generate_clicks = []

# 3) Placeholder _after_ the input & button
result_placeholder = st.empty()

# 4) Generation logic
def generate(prompt: str):
    if not prompt.strip():
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

# 5) If either Enter or Button was used, run generation
if st.session_state.generate_clicks:
    # pop so we only handle once
    st.session_state.generate_clicks.pop()
    generate(st.session_state.prompt)
