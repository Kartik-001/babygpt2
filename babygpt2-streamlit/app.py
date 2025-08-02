# app.py
import streamlit as st
import requests

# 0️⃣ Page setup
st.set_page_config(page_title="BabyGPT2 Chat", page_icon="🤖")
st.title("🤖 BabyGPT2 Chat")
st.markdown("""
Enter a prompt below and either **press Enter** or click **Generate**.  
Your text will be sent to the BabyGPT2 API and its continuation will appear below.
""")

# 1️⃣ Create a placeholder where both Enter and Button will render output
result_placeholder = st.empty()

# 2️⃣ Define the generate callback to write into that placeholder
def generate():
    prompt = st.session_state.prompt.strip()
    if not prompt:
        # clear any previous content and show a warning
        result_placeholder.empty()
        result_placeholder.warning("Please enter a prompt first.")
        return

    # everything below runs inside the placeholder
    with result_placeholder:
        with st.spinner("Generating..."):
            resp = requests.post(
                "https://babygpt2-api.onrender.com/generate",
                json={"prompt": prompt}
            )
            resp.raise_for_status()
            text = resp.json().get("text", "")

        # display result in the same spot
        st.subheader("🖋️ Generated Text")
        st.write(text)

# 3️⃣ Single-line input that fires on Enter
st.text_input(
    "Your prompt (press Enter to send)",
    key="prompt",
    on_change=generate
)

# 4️⃣ Also keep the Generate button
if st.button("Generate"):
    generate()
