
"""API Olfactive Studio : PostgreSQL + ChromaDB + Ollama."""

import json
import unicodedata

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

import config
import rag
from database import get_connection


app = FastAPI(
    title="Olfactive Studio API",
    description="Assistant aromatique conversationnel avec RAG.",
    version="4.1.0",
)


# --------------------------------------------------
# MODÈLES
# --------------------------------------------------

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(config.TOP_K, ge=1, le=20)
    history: list[dict] = Field(default_factory=list)


class AromaSource(BaseModel):
    source_type: str
    source_id: str
    is_synthetic: bool
    text: str
    score: float = Field(
        ...,
        description="Similarité cosinus (1 = identique)",
    )


# --------------------------------------------------
# NORMALISATION
# --------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def get_conversation_text(
    question: str,
    history: list[dict],
) -> str:
    messages = []

    for item in history[-8:]:
        if (
            isinstance(item, dict)
            and item.get("role") == "user"
            and isinstance(item.get("content"), str)
        ):
            messages.append(item["content"])

    messages.append(question)

    return " ".join(messages)


# --------------------------------------------------
# DÉTECTION DES PRÉFÉRENCES
# --------------------------------------------------

def has_sensory_preference(text: str) -> bool:
    q = normalize(text)

    preferences = [
        "acidule",
        "acide",
        "doux",
        "sucre",
        "amer",
        "fruite",
        "intense",
        "leger",
        "puissant",
        "frais",
        "floral",
        "epice",
    ]

    return any(word in q for word in preferences)


# --------------------------------------------------
# DÉTECTION DES CRITÈRES DU CATALOGUE
# --------------------------------------------------

def detect_family(question: str) -> str | None:
    q = normalize(question)

    families = {
        "agrum": "Agrume",
        "fruit rouge": "Fruit rouge",
        "fruits rouges": "Fruit rouge",
        "fruit exotique": "Fruit exotique",
        "fruits exotiques": "Fruit exotique",
        "gourmand": "Gourmande",
        "floral": "Florale",
        "fleur": "Florale",
        "epice": "Épicée",
        "menthol": "Mentholée",
    }

    for keyword, family in families.items():
        if keyword in q:
            return family

    return None


def detect_application(question: str) -> str | None:
    q = normalize(question)

    applications = {
        "boisson": "boisson",
        "biscuit": "biscuit",
        "confiserie": "confiserie",
        "dessert": "dessert",
        "yaourt": "yaourt",
        "produit laitier": "laitier",
        "glace": "glace",
    }

    for keyword, application in applications.items():
        if keyword in q:
            return application

    return None


# --------------------------------------------------
# RECHERCHE POSTGRESQL
# --------------------------------------------------

def search_catalogue(
    question: str,
    top_k: int,
) -> tuple[list[dict], str | None, str | None]:

    family = detect_family(question)
    application = detect_application(question)

    if not family and not application:
        return [], None, None

    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = "SELECT * FROM aromes WHERE 1=1"
            params = []

            if family:
                query += " AND famille ILIKE %s"
                params.append(f"%{family}%")

            if application:
                query += " AND applications ILIKE %s"
                params.append(f"%{application}%")

            query += " ORDER BY nom LIMIT %s"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows], family, application

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur PostgreSQL : {exc}",
        ) from exc

    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------
# CONVERSATION AVEC LE LLM
# --------------------------------------------------

def generate_client_guidance(
    question: str,
    history: list[dict],
    rows: list[dict],
) -> dict:

    catalogue = "\n".join(
        f"- {row['nom']} : famille {row['famille']}, "
        f"notes {row['notes']}, "
        f"applications {row['applications']}"
        for row in rows
    )

    if not catalogue:
        catalogue = "Aucun candidat sélectionné pour le moment."

    messages = [
        {
            "role": "system",
            "content": """
Tu es le conseiller aromatique d'Olfactive Studio.

Ton rôle est de guider le client pour comprendre son besoin
avant de présenter les arômes disponibles.

Tu dois progressivement identifier :
- Le produit alimentaire concerné.
- La famille aromatique recherchée.
- Le profil sensoriel souhaité.
- Les contraintes techniques éventuelles.
- Le budget si nécessaire.

RÈGLES :
1. Analyse toute la conversation.
2. Ne redemande jamais une information déjà donnée.
3. Pose une seule question à la fois.
4. Privilégie les questions utiles.
5. Ne suppose jamais les préférences du client.
6. N'invente aucune caractéristique.
7. Ne recommande aucun arôme absent du catalogue.
8. Si le besoin est suffisamment précis, indique que
   les candidats peuvent être présentés.
9. Réponds naturellement et en français.

Retourne obligatoirement un JSON valide, sans Markdown.

Si une information essentielle manque :
{
  "status": "needs_clarification",
  "question": "Une seule question à poser"
}

Si le besoin est suffisamment précis :
{
  "status": "recommendations_ready",
  "question": ""
}
"""
        }
    ]

    for item in history[-8:]:
        if (
            isinstance(item, dict)
            and item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        ):
            messages.append({
                "role": item["role"],
                "content": item["content"],
            })

    messages.append({
        "role": "user",
        "content": (
            f"Nouvelle demande : {question}\n\n"
            f"Candidats du catalogue :\n{catalogue}\n\n"
            "Analyse la conversation et retourne uniquement le JSON."
        ),
    })

    try:
        response = rag.chat(messages)

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("JSON absent")

        result = json.loads(response[start:end + 1])

        status = result.get("status")
        clarification = result.get("question", "")

        if status == "needs_clarification":
            if not isinstance(clarification, str) or not clarification.strip():
                raise ValueError("Question absente")

            return {
                "status": "needs_clarification",
                "question": clarification.strip(),
            }

        if status == "recommendations_ready":
            return {
                "status": "recommendations_ready",
                "question": "",
            }

        raise ValueError("Statut invalide")

    except Exception:
        return {
            "status": "needs_clarification",
            "question": (
                "Pouvez-vous préciser le type de produit alimentaire "
                "dans lequel vous souhaitez utiliser cet arôme ?"
            ),
        }


# --------------------------------------------------
# CONSTRUCTION DES SOURCES
# --------------------------------------------------

def build_source(row: dict) -> AromaSource:
    text = (
        f"Arôme : {row['nom']}. "
        f"Famille : {row['famille']}. "
        f"Notes : {row['notes']}. "
        f"Applications : {row['applications']}. "
        f"Sucrosité : {row['sucrosite']}. "
        f"Tenue à la chaleur : {row['tenue_chaleur']}. "
        f"Certifications : {row['certifications']}. "
        f"Prix indicatif : {row['prix_indicatif_eur_kg']} euros/kg. "
        f"Délai : {row['delai_jours']} jours. "
        f"Volume minimum : {row['volume_minimum_kg']} kg."
    )

    return AromaSource(
        source_type="arome",
        source_id=str(row["id"]),
        is_synthetic=bool(row["is_synthetic"]),
        text=text,
        score=1.0,
    )


# --------------------------------------------------
# FORMATAGE DES RECOMMANDATIONS
# --------------------------------------------------

def format_catalogue_answer(
    rows: list[dict],
    family: str | None,
    application: str | None,
) -> str:

    if not rows:
        criteria = []

        if family:
            criteria.append(f"la famille {family}")

        if application:
            criteria.append(f"l'application {application}")

        if not criteria:
            return (
                "Je n'ai trouvé aucun candidat correspondant "
                "dans le catalogue."
            )

        return (
            "Je n'ai trouvé aucun arôme correspondant à "
            + " et ".join(criteria)
            + " dans le catalogue."
        )

    lines = [
        "Voici les arômes correspondant aux critères "
        "identifiés dans le catalogue :"
    ]

    for index, row in enumerate(rows, start=1):
        lines.extend([
            "",
            f"### {index}. {row['nom']}",
            f"- Famille : {row['famille']}",
            f"- Notes : {row['notes']}",
            f"- Applications : {row['applications']}",
            f"- Sucrosité : {row['sucrosite']}",
            f"- Tenue à la chaleur : {row['tenue_chaleur']}",
            f"- Certifications : {row['certifications']}",
            f"- Prix indicatif : {row['prix_indicatif_eur_kg']} €/kg",
            f"- Délai : {row['delai_jours']} jours",
            f"- Volume minimum : {row['volume_minimum_kg']} kg",
        ])

    if any(bool(row["is_synthetic"]) for row in rows):
        lines.extend([
            "",
            "**Attention :** les données présentées sont fictives "
            "et servent à la démonstration.",
        ])

    if application:
        lines.extend([
            "",
            "Les applications indiquées proviennent du catalogue. "
            "Elles ne garantissent pas à elles seules la pertinence "
            "sensorielle pour votre recette.",
        ])

    return "\n".join(lines)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/health", tags=["Système"])
def health():
    try:
        count = rag.get_collection().count()
    except Exception:
        count = None

    return {
        "status": "online",
        "service": "Olfactive Studio",
        "documents_indexed": count,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/aromes", tags=["Catalogue aromatique"])
def get_aromes(
    famille: str | None = None,
    application: str | None = None,
):
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
            return [dict(row) for row in cursor.fetchall()]

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de lire le catalogue : {exc}",
        ) from exc

    finally:
        if conn is not None:
            conn.close()


@app.post(
    "/search",
    response_model=list[AromaSource],
    tags=["RAG aromatique"],
)
def search(req: SearchRequest):
    try:
        documents = rag.search_documents(
            req.question,
            req.top_k,
        )

        return [
            AromaSource(
                source_type=document["metadata"]["source_type"],
                source_id=str(document["metadata"]["source_id"]),
                is_synthetic=(
                    str(document["metadata"]["is_synthetic"]).lower()
                    == "true"
                ),
                text=document["text"],
                score=document["score"],
            )
            for document in documents
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur ChromaDB : {exc}",
        ) from exc


@app.post("/ask", tags=["Assistant conversationnel"])
def ask(req: SearchRequest):

    conversation = get_conversation_text(
        req.question,
        req.history,
    )

    rows, family, application = search_catalogue(
        conversation,
        req.top_k,
    )

    # Première étape : demander le profil sensoriel.
    if not has_sensory_preference(conversation):
        return {
            "question": req.question,
            "status": "needs_clarification",
            "answer": (
                "Quel profil aromatique recherchez-vous : "
                "plutôt acidulé et intense, ou doux et fruité ?"
            ),
            "model": config.LLM_MODEL,
            "sources": [],
        }

    # Deuxième étape : laisser le LLM identifier les autres
    # informations manquantes.
    guidance = generate_client_guidance(
        req.question,
        req.history,
        rows,
    )

    if guidance["status"] == "needs_clarification":
        return {
            "question": req.question,
            "status": "needs_clarification",
            "answer": guidance["question"],
            "model": config.LLM_MODEL,
            "sources": [],
        }

    # Troisième étape : présenter les données du catalogue.
    sources = [
        build_source(row)
        for row in rows
    ]

    answer = format_catalogue_answer(
        rows,
        family,
        application,
    )

    return {
        "question": req.question,
        "status": "recommendations_ready",
        "answer": answer,
        "model": config.LLM_MODEL,
        "sources": [
            source.model_dump()
            for source in sources
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
