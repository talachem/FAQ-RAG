import numpy as np
from ollama import chat, embeddings
import json, os
import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="ChatFAQ")

conversationOllama = [
        {
            "role": "system",
            "content": """
                Du bist ein hilfsbereiter und freundlicher ChatBot, der Kundenanfragen beantwortet.
                Nutze die bereitgestellten FAQs als Kontext, um präzise und verständliche Antworten zu geben.
                Wenn eine Frage nicht direkt beantwortet werden kann, versuche, relevante Informationen aus den FAQs zu nutzen,
                um eine hilfreiche Antwort zu formulieren.
                """
        }
]


def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if len(vault_embeddings) == 0:
        return []

    # Get embedding for input
    input_embedding = np.array(embeddings(model='embeddinggemma', prompt=rewritten_input)["embedding"])

    # Normalize vault and input embeddings
    vault_embeddings = np.array(vault_embeddings)
    input_norm = np.linalg.norm(input_embedding)
    vault_norms = np.linalg.norm(vault_embeddings, axis=1)

    # Compute cosine similarity
    cos_scores = np.dot(vault_embeddings, input_embedding) / (vault_norms * input_norm + 1e-10)

    # Get top-k indices sorted by similarity
    top_k = min(top_k, len(cos_scores))
    top_indices = np.argsort(cos_scores)[-top_k:][::-1]

    # Retrieve corresponding text
    relevant_context = [vault_content[idx] for idx in top_indices]
    return relevant_context


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        faqs = get_relevant_context("was ist smartdeal", embeddedQuestions, fullFAQs, top_k=4)

        for faq in faqs:
            conversationOllama.append(
                {
                    "role": "user",
                    "content": f"Frage: \"{faq['question']}\", Antwort: \"{faq['answer']}\""
                }
            )
        
        conversationOllama.append(
            {"role": "user", "content": user_text}
        )
        st.session_state["messages"].append((user_text, True))
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            response = chat(model="gemma3:4b", messages=conversationOllama)
        
        conversationOllama.append(
            {"role": "assistant", "content": response.message.content}
        )
        st.session_state["messages"].append((response.message.content, False))
        st.session_state["user_input"] = ""


def loadFAQ():
    if os.path.exists("FAQ.json"):
        with open("FAQ.json", 'r', encoding='utf-8') as f:
            vaultContent = json.load(f)

    vault_embeddings = []
    questions = [item["question"] for item in vaultContent if len(item['question']) > 0]
    for content in questions:
        response = embeddings(model='embeddinggemma', prompt=content)
        vault_embeddings.append(response["embedding"])

    return vaultContent, vault_embeddings


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []

    st.header("FAQ Chat")
    st.session_state["thinking_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    fullFAQs, embeddedQuestions = loadFAQ()
    page()