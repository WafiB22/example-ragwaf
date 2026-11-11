
"""Moteur RAG spécialisé dans les arômes et les saveurs."""

from __future__ import annotations

import chromadb
import ollama

import config

_ollama = ollama.Client(host=config.OLLAMA_HOST)

SYSTEM_PROMPT = """
Tu es Olfactive Studio, un assistant expert en arômes et saveurs
destiné aux professionnels de l'industrie alimentaire.

Ton rôle :
- Comprendre les descriptions sensorielles des clients.
- Identifier les familles aromatiques et les notes pertinentes.
- Expliquer les caractéristiques des arômes.
- Proposer des pistes de création adaptées aux applications demandées.

Règles :
- Utilise prioritairement les informations des documents retrouvés.
- Ne prétends jamais connaître une information absente des documents.
- Ne fabrique pas de certifications, de prix ou de caractéristiques techniques.
- Les données du catalogue de démonstration sont synthétiques.
- Ne présente jamais une suggestion comme une formule industrielle validée.
- Les dosages, la sécurité et la conformité doivent être validés par le laboratoire.
- Réponds en français, de manière claire et professionnelle.
"""


def get_chroma_client() -> chromadb.ClientAPI:
    """Ouvre la base vectorielle persistante."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI | None = None):
    """Récupère ou crée la collection documentaire aromatique."""
    client = client or get_chroma_client()

    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": config.EMBED_MODEL,
        },
    )


def embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Calcule les embeddings avec Ollama."""
    inputs = [prefix + text for text in texts]

    response = _ollama.embed(
        model=config.EMBED_MODEL,
        input=inputs,
    )

    return response["embeddings"]


def _uses_nomic_prefixes() -> bool:
    return "nomic" in config.EMBED_MODEL.lower()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Transforme les documents en vecteurs."""
    prefix = "search_document: " if _uses_nomic_prefixes() else ""
    return embed(texts, prefix)


def embed_query(text: str) -> list[float]:
    """Transforme une question client en vecteur."""
    prefix = "search_query: " if _uses_nomic_prefixes() else ""
    return embed([text], prefix)[0]


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Recherche les documents aromatiques les plus pertinents."""

    collection = get_collection()

    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = []

    for doc, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        documents.append({
            "text": doc,
            "metadata": metadata,
            "score": round(1 - distance, 4),
        })

    return documents


def chat(messages: list[dict]) -> str:
    """Envoie une conversation au modèle Ollama."""

    response = _ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        options={"temperature": 0.2},
    )

    return response["message"]["content"]


def answer_question(question: str, top_k: int = 5) -> dict:
    """Recherche les connaissances et génère une réponse aromatique."""

    documents = search_documents(question, top_k)

    if not documents:
        return {
            "question": question,
            "answer": (
                "Je ne dispose pas encore de documentation aromatique "
                "suffisante pour répondre à cette demande."
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        document["text"] for document in documents
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Documentation aromatique retrouvée :\n\n{context}\n\n"
                f"Demande du client : {question}\n\n"
                "Réponds en t'appuyant sur cette documentation."
            ),
        },
    ]

    answer = chat(messages)

    return {
        "question": question,
        "answer": answer.strip(),
        "sources": documents,
    }
