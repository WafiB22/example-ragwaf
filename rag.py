
"""Gestion du RAG avec Ollama et ChromaDB."""

import chromadb
import ollama

import config


SYSTEM_PROMPT = """
Tu es Olfactive Studio, un assistant expert en arômes alimentaires.

Tu aides les clients à préciser leur besoin et à comprendre
les informations disponibles dans le catalogue.

RÈGLES :
1. Utilise uniquement les informations fournies.
2. N'invente jamais de prix, délais, certifications ou caractéristiques.
3. Ne suppose jamais le budget du client.
4. Ne confonds pas fraîcheur sensorielle et tenue à la chaleur.
5. Ne présente jamais une recommandation comme une formule
   industrielle validée.
6. Les données is_synthetic=true sont fictives et doivent être
   signalées lorsqu'elles sont présentées.
7. Si une information essentielle manque, pose une question
   courte et précise.
8. Pose une seule question à la fois.
9. Ne redemande pas une information déjà donnée.
10. Réponds en français avec un ton professionnel et naturel.
"""


def get_chroma_client():
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_documents(documents: list[str]) -> list[list[float]]:
    embeddings = []

    for document in documents:
        response = ollama.embeddings(
            model=config.EMBED_MODEL,
            prompt=document,
        )
        embeddings.append(response["embedding"])

    return embeddings


def embed_query(question: str) -> list[float]:
    response = ollama.embeddings(
        model=config.EMBED_MODEL,
        prompt=question,
    )
    return response["embedding"]


def search_documents(
    question: str,
    top_k: int | None = None,
) -> list[dict]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k or config.TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = []

    for index, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        documents.append({
            "text": document,
            "metadata": metadata,
            "score": max(0.0, 1.0 - distance),
        })

    return documents


def chat(messages: list[dict]) -> str:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        think=False,
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()


def answer_question(
    question: str,
    top_k: int | None = None,
) -> str:
    documents = search_documents(question, top_k)

    if not documents:
        return (
            "Je ne dispose pas de données suffisantes "
            "pour répondre à cette demande."
        )

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
                f"Informations disponibles :\n{context}\n\n"
                f"Question : {question}\n\n"
                "Réponds uniquement à partir des informations fournies."
            ),
        },
    ]

    return chat(messages)
