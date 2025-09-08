import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from app import indexing, query

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

indexing.main()


def simple_chat(message: str, history: list[tuple[str, str]]) -> str:
    """
    Handle a conversational turn with memory using the OpenAI chat model.
    Args:
        message (str): The latest user message.
        history (list[tuple[str, str]]): Conversation history where each tuple
            is (user_message, assistant_response).

    Returns:
        str: The assistant's response to the current user message.
    """
    context = "Tu es un assistant médical qui est entraîné à répondre à des questions sur les pneumopathies communautaires.\n\n"
    messages = [{"role": "system", "content": context}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    prompt = query.build_prompt(message)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    response_text = response.choices[0].message.content
    return response_text  # type: ignore


with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Medical RAG Chatbot")
    gr.ChatInterface(fn=simple_chat)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
