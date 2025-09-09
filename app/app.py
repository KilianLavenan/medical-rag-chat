import os
from pathlib import Path

import indexing
import query
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺", layout="centered")


@st.cache_resource  # type: ignore
def init_indexing() -> bool:
    try:
        indexing.main()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation: {e}")
        return False


def generate_response(message: str, history: list[dict[str, str]]) -> str:
    """
    Handle a conversational turn with memory using the OpenAI chat model.
    Args:
        message (str): The latest user message.
        history (list[dict[str, str]]): Conversation history where each dict
            contains "user_message" and "assistant_response".

    Returns:
        str: The assistant's response to the current user message.
    """
    context = "Tu es un assistant médical qui est entraîné à répondre à des questions sur les pneumopathies communautaires.\n\n"
    messages = [{"role": "system", "content": context}]
    messages.extend(history)
    prompt = query.build_prompt(message)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    response_text = response.choices[0].message.content
    return response_text  # type: ignore


def main() -> None:
    st.title("🩺 Medical RAG Chatbot")
    st.markdown("Assistant médical spécialisé dans les pneumopathies communautaires")
    chroma_db_path = Path("data/chroma_db")
    if chroma_db_path.exists():
        st.session_state.indexing_initialized = True

    if "indexing_initialized" not in st.session_state:
        with st.spinner("Initialisation du système RAG..."):
            st.session_state.indexing_initialized = init_indexing()

    if not st.session_state.indexing_initialized:
        st.error("Impossible d'initialiser le système. Veuillez recharger la page.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Bonjour ! Je suis votre assistant médical spécialisé dans les pneumopathies communautaires. Comment puis-je vous aider ?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la base de connaissances..."):
                response = generate_response(prompt, st.session_state.messages[:-1])
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
