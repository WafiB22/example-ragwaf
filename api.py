
"""API Olfactive Studio : catalogue aromatique + RAG pédagogique.

Lancement :
    uvicorn api:app --reload

Swagger :
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

import config
import rag
from database import get_connection


SYSTEM_PROMPT = """Tu es un conteur bienveillant qui répond aux questions
sur une histoire.
Règles :
- Réponds uniquement à partir des extraits fournis.
- Si la réponse ne se trouve pas dans les extraits, indique-le.
- Réponds en français, clairement et brièvement."""


app = FastAPI(
    title="Olfactive Studio API",
    description=(
        "API de démonstration : catalogue aromatique PostgreSQL "
        "et recherche documentaire RAG avec Ollama."
    ),
    version="1.1.0",
)


# --------------------------------------------------
# Modèles utilisés par le RAG pédagogique
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["Comment s'appelle le cheval de Loran ?"]
    )
    top_k: int = Field(
        config.TOP_K,
        ge=1,
        le=20,
        description="Nombre de passages à retrouver"
    )


class Source(BaseModel):
    chapter: str
    text: str
    score: float = Field(description="Similarité cosinus (1 = identique)")


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    sources: list[Source]


# --------------------------------------------------
# Recherche dans ChromaDB
# --------------------------------------------------

def retrieve(question: str, top_k: int) -> list[Source]:
    collection = rag.get_collection()

    if collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="La base vectorielle est vide. Lance python ingest.py."
        )

    result = collection.query(
        query_embeddings=[rag.embed_query(question)],
        n_results=min(top_k, collection.count()),
    )

    return [
        Source(
            chapter=meta["chapter"],
            text=doc,
            score=round(1 - dist, 4)
        )
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        )
    ]


# --------------------------------------------------
# Route système
# --------------------------------------------------

@app.get("/health", tags=["Système"])
def health():
    """Vérifie l'état du service et des modèles."""

    try:
        chunks = rag.get_collection().count()
    except Exception:
        chunks = None

    return {
        "status": "online",
        "service": "Olfactive Studio",
        "chunks_rag": chunks,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


# --------------------------------------------------
# Catalogue aromatique PostgreSQL
# --------------------------------------------------

@app.get("/aromes", tags=["Olfactive Studio"])
def get_aromes(
    famille: str | None = None,
    application: str | None = None
):
    """
    Retourne le catalogue aromatique.

    Filtres facultatifs :
    - famille : recherche dans la famille aromatique
    - application : recherche dans les applications
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = "SELECT * FROM aromes WHERE 1=1"
            params = []

            if famille:
                query += " AND famille ILIKE %s"
                params.append(f"%{famille}%")

            if application:
                query += " AND applications ILIKE %s"
                params.append(f"%{application}%")

            query += " ORDER BY nom"

            cursor.execute(query, params)
            aromes = cursor.fetchall()

        return [dict(arome) for arome in aromes]

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de lire le catalogue aromatique : {exc}"
        ) from exc

    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------
# Routes RAG pédagogiques conservées
# --------------------------------------------------

@app.post("/search", response_model=list[Source], tags=["RAG pédagogique"])
def search(req: AskRequest):
    """Retourne les passages pertinents sans appeler le LLM."""
    return retrieve(req.question, req.top_k)


@app.post("/ask", response_model=AskResponse, tags=["RAG pédagogique"])
def ask(req: AskRequest):
    """Répond à une question à partir des passages retrouvés."""

    sources = retrieve(req.question, req.top_k)

    context = "\n\n---\n\n".join(
        f"[{source.chapter}]\n{source.text}"
        for source in sources
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Extraits de l'histoire :\n\n{context}\n\n"
                f"Question : {req.question}"
            )
        },
    ]

    try:
        answer = rag.chat(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur Ollama : {exc}"
        ) from exc

    return AskResponse(
        question=req.question,
        answer=answer.strip(),
        model=config.LLM_MODEL,
        sources=sources
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
