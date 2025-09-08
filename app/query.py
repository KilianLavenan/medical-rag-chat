from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


def query_chroma(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    """
    Query the Chroma vector database with a natural language question.

    The function encodes the input query into an embedding using a
    SentenceTransformer model, then searches the Chroma collection for
    the most semantically similar document chunks.

    Args:
        query (str): The user question or search query.
        n_results (int, optional): Number of most relevant chunks to retrieve.
            Defaults to 3.

    Returns:
        list[dict]: Query results from Chroma
    """
    chroma_db_path = Path("data/chroma_db")
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_collection(name="medical_rag")

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results  # type: ignore


def build_prompt(query: str) -> str:
    results = query_chroma(query=query)
    retrieved_chunks = results["documents"][0]  # type: ignore
    prompt = "Voici quelques informations du document de prise en charge des pneumopathies aigues communautaires qui pourraient être utiles pour répondre aux questions de l'utilisateur:\n---\n"
    for chunk in retrieved_chunks:
        prompt += chunk + "\n\n"

    prompt += f"\nUser Question: {query}\n\n"
    prompt += (
        "Ta réponse doit OBLIGATOIREMENT suivre ce format :\n\n"
        "Réponse : (fournis une réponse complète et synthétisée basée UNIQUEMENT sur les informations ci-dessus)\n\n"
        "Source : (Liste les passages exacts et leur parties dans le texte utilisés pour répondre. "
        "Si le passage provient d’un tableau ou d’une annexe, inclue également son titre, par exemple : "
        "Si tu ne trouves pas la réponse dans les informations fournies, ne réponds pas à la question,"
        "tu peux dire que tu n'as pas la réponse."
    )
    return prompt
