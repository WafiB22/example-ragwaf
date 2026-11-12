
"""API Olfactive Studio : PostgreSQL + ChromaDB + Ollama."""

import json
import logging
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

import config
import rag
from database import get_connection


# --------------------------------------------------
# JOURNALISATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("olfactive_studio")


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Olfactive Studio API",
    description="Assistant aromatique conversationnel avec RAG.",
    version="6.2.0",
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
        description="Score associé à la source",
    )


# --------------------------------------------------
# NORMALISATION
# --------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())

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
# DÉTECTION DES PRÉFÉRENCES SENSORIELLES
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
        "floral",
        "epice",
        "sale",
        "gourmand",
        "frais",
    ]

    return any(word in q for word in preferences)


# --------------------------------------------------
# DÉTECTION DE LA FAMILLE
# --------------------------------------------------

def detect_family(question: str) -> str | None:
    q = normalize(question)

    families = {
        "fruits rouges": "Fruits rouges",
        "fruit rouge": "Fruits rouges",
        "fraise": "Fruits rouges",
        "framboise": "Fruits rouges",
        "cerise": "Fruits rouges",

        "fruits tropicaux": "Fruits tropicaux",
        "fruit tropical": "Fruits tropicaux",
        "fruit exotique": "Fruits tropicaux",
        "fruits exotiques": "Fruits tropicaux",
        "exotique": "Fruits tropicaux",
        "mangue": "Fruits tropicaux",
        "ananas": "Fruits tropicaux",
        "passion": "Fruits tropicaux",

        "agrum": "Agrumes",
        "citron": "Agrumes",
        "orange": "Agrumes",
        "pamplemousse": "Agrumes",
        "mandarine": "Agrumes",

        "gourmand": "Notes gourmandes",
        "vanille": "Notes gourmandes",
        "caramel": "Notes gourmandes",
        "chocolat": "Notes gourmandes",
        "biscuit": "Notes gourmandes",

        "epice": "Herbes et épices",
        "herbe": "Herbes et épices",
        "gingembre": "Herbes et épices",
        "cardamome": "Herbes et épices",
        "cannelle": "Herbes et épices",
        "poivre": "Herbes et épices",

        "sale": "Notes salées",
        "umami": "Notes salées",
        "bouillon": "Notes salées",
    }

    for keyword in sorted(families, key=len, reverse=True):
        if keyword in q:
            return families[keyword]

    return None


# --------------------------------------------------
# DÉTECTION DE L'APPLICATION
# --------------------------------------------------

def detect_application(question: str) -> str | None:
    q = normalize(question)

    applications = {
        "boisson": "boissons",
        "biscuit": "biscuits",
        "confiserie": "confiserie",
        "dessert": "desserts",
        "yaourt": "produits laitiers",
        "produit laitier": "produits laitiers",
        "glace": "glaces",
    }

    for keyword in sorted(applications, key=len, reverse=True):
        if keyword in q:
            return applications[keyword]

    return None


# --------------------------------------------------
# DÉTECTION DE LA SUCROSITÉ
# --------------------------------------------------

def detect_sweetness_filter(question: str) -> list[dict]:
    q = normalize(question)

    def make_filter(operator: str, value: int) -> list[dict]:
        return [{
            "column": "sucrosite",
            "label": "sucrosité",
            "operator": operator,
            "value": Decimal(str(value)),
        }]

    # Niveau 1 : absence ou très faible sucrosité
    very_low = [
        "sans sucre",
        "non sucre",
        "pas sucre",
        "tres peu sucre",
        "extremement peu sucre",
        "sucrosite minimale",
        "sucrosite de 1",
        "niveau de sucre 1",
    ]

    # Niveaux 1 à 3 : faible sucrosité
    low = [
        "pas trop sucre",
        "peu sucre",
        "faiblement sucre",
        "moins sucre",
        "peu de sucre",
        "faible sucrosite",
        "legerement sucre",
        "sucrosite faible",
    ]

    # Niveau 3 : sucrosité moyenne
    medium = [
        "moyennement sucre",
        "equilibre en sucre",
        "sucrosite moyenne",
        "niveau de sucre moyen",
        "sucrosite de 3",
        "niveau de sucre 3",
    ]

    # Niveaux 4 à 5 : forte sucrosité
    high = [
        "bien sucre",
        "tres sucre",
        "fortement sucre",
        "beaucoup de sucre",
        "forte sucrosite",
        "arome sucre",
        "arome tres sucre",
        "plutot sucre",
        "plus sucre",
        "sucrosite elevee",
        "sucrosite forte",
        "sucrosite de 4",
        "niveau de sucre 4",
    ]

    # Niveau 5 : sucrosité maximale
    very_high = [
        "extremement sucre",
        "tres fortement sucre",
        "tres riche en sucre",
        "sucrosite maximale",
        "sucrosite de 5",
        "niveau de sucre 5",
    ]

    # Priorité aux expressions les plus précises
    if any(expression in q for expression in very_low):
        return make_filter("<=", 1)

    if any(expression in q for expression in very_high):
        return make_filter("=", 5)

    if any(expression in q for expression in medium):
        return make_filter("=", 3)

    if any(expression in q for expression in low):
        return make_filter("<=", 3)

    if any(expression in q for expression in high):
        return make_filter(">=", 4)

    # Détection générique d'une demande sucrée
    if re.search(r"\bsucre\b", q):
        return make_filter(">=", 4)

    return []


# --------------------------------------------------
# DÉTECTION DE LA TENUE À LA CHALEUR
# --------------------------------------------------

def detect_heat_filter(question: str) -> str | None:
    q = normalize(question)

    heat_keywords = [
        "pasteurise",
        "pasteurisation",
        "cuisson",
        "haute temperature",
        "haute chaleur",
        "resistant a la chaleur",
        "resistance a la chaleur",
        "tenue a la chaleur",
    ]

    if any(keyword in q for keyword in heat_keywords):
        return "élevée"

    return None


# --------------------------------------------------
# DÉTECTION DES CRITÈRES NUMÉRIQUES
# --------------------------------------------------

NUMERIC_FIELDS = {
    "sucrosite": {
        "column": "sucrosite",
        "label": "sucrosité",
        "aliases": ["sucrosite"],
    },
    "prix_indicatif_eur_kg": {
        "column": "prix_indicatif_eur_kg",
        "label": "prix",
        "aliases": [
            "prix au kilo",
            "prix par kilo",
            "prix au kg",
            "prix par kg",
            "prix",
        ],
    },
    "delai_jours": {
        "column": "delai_jours",
        "label": "délai",
        "aliases": ["delai", "delais"],
    },
    "volume_minimum_kg": {
        "column": "volume_minimum_kg",
        "label": "volume minimum",
        "aliases": [
            "volume minimum",
            "volume minimal",
            "quantite minimale",
            "quantite minimum",
        ],
    },
}


OPERATORS = [
    (r">=", ">="),
    (r"<=", "<="),
    (r">", ">"),
    (r"<", "<"),
    (r"=", "="),
    (
        r"(?:superieur(?:e)?\s+ou\s+egal(?:e)?\s+a"
        r"|au\s+moins|minimum\s+de)",
        ">=",
    ),
    (
        r"(?:inferieur(?:e)?\s+ou\s+egal(?:e)?\s+a"
        r"|au\s+plus|maximum\s+de)",
        "<=",
    ),
    (
        r"(?:strictement\s+)?superieur(?:e)?\s+a"
        r"|plus\s+de|au-dessus\s+de",
        ">",
    ),
    (
        r"(?:strictement\s+)?inferieur(?:e)?\s+a"
        r"|moins\s+de|en-dessous\s+de",
        "<",
    ),
    (
        r"(?:egal(?:e)?\s+a|exactement)",
        "=",
    ),
]


def detect_numeric_filters(question: str) -> list[dict]:
    q = normalize(question)
    filters = []

    for field in NUMERIC_FIELDS.values():
        aliases = sorted(
            field["aliases"],
            key=len,
            reverse=True,
        )

        for alias in aliases:
            alias_pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            for operator_pattern, sql_operator in OPERATORS:
                pattern = (
                    alias_pattern
                    + r"\s*(?:est\s*)?"
                    + r"(?:"
                    + operator_pattern
                    + r")"
                    + r"\s*(-?\d+(?:[.,]\d+)?)"
                )

                match = re.search(pattern, q)

                if not match:
                    continue

                raw_value = match.group(1).replace(",", ".")

                try:
                    value = Decimal(raw_value)
                except InvalidOperation:
                    continue

                filters.append({
                    "column": field["column"],
                    "label": field["label"],
                    "operator": sql_operator,
                    "value": value,
                })

                break

            if any(
                item["column"] == field["column"]
                for item in filters
            ):
                break

    return filters


# --------------------------------------------------
# RECHERCHE POSTGRESQL
# --------------------------------------------------

def search_catalogue(
    question: str,
    top_k: int,
) -> tuple[list[dict], str | None, str | None, list[dict]]:

    family = detect_family(question)
    application = detect_application(question)
    numeric_filters = detect_numeric_filters(question)

    sweetness_filters = detect_sweetness_filter(question)

    # Un filtre numérique explicitement demandé est prioritaire.
    if not any(
        item["column"] == "sucrosite"
        for item in numeric_filters
    ):
        numeric_filters.extend(sweetness_filters)

    heat_filter = detect_heat_filter(question)

    if not any([
        family,
        application,
        numeric_filters,
        heat_filter,
    ]):
        return [], None, None, []

    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = "SELECT * FROM public.food_aromas WHERE 1=1"
            params = []

            # Filtre famille
            if family:
                query += " AND famille ILIKE %s"
                params.append(family)

            # Filtre application
            if application:
                query += """
                    AND EXISTS (
                        SELECT 1
                        FROM unnest(applications) AS app
                        WHERE app ILIKE %s
                    )
                """
                params.append(f"%{application}%")

            # Filtres numériques
            allowed_columns = {
                "sucrosite",
                "prix_indicatif_eur_kg",
                "delai_jours",
                "volume_minimum_kg",
            }

            allowed_operators = {
                ">", "<", ">=", "<=", "=",
            }

            for criterion in numeric_filters:
                column = criterion["column"]
                operator = criterion["operator"]

                if (
                    column not in allowed_columns
                    or operator not in allowed_operators
                ):
                    continue

                query += f" AND {column} {operator} %s"
                params.append(criterion["value"])

            # Filtre strict de tenue à la chaleur
            if heat_filter:
                query += " AND tenue_chaleur = %s"
                params.append(heat_filter)

            # Classement selon la préférence de sucrosité
            order_parts = []

            if heat_filter:
                order_parts.append("""
                    CASE
                        WHEN tenue_chaleur = 'élevée' THEN 0
                        ELSE 1
                    END
                """)

            sweetness_criterion = next(
                (
                    item for item in numeric_filters
                    if item["column"] == "sucrosite"
                ),
                None,
            )

            if sweetness_criterion:
                operator = sweetness_criterion["operator"]

                if operator in (">", ">="):
                    order_parts.append(
                        "sucrosite DESC NULLS LAST"
                    )
                else:
                    order_parts.append(
                        "sucrosite ASC NULLS LAST"
                    )

            order_parts.append("nom ASC")

            query += " ORDER BY " + ", ".join(order_parts)
            query += " LIMIT %s"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return (
            [dict(row) for row in rows],
            family,
            application,
            numeric_filters,
        )

    except Exception as exc:
        logger.exception("Erreur pendant la recherche PostgreSQL")

        raise HTTPException(
            status_code=503,
            detail="Erreur lors de la recherche dans PostgreSQL.",
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

Les familles du catalogue sont exclusivement :
- Agrumes
- Fruits rouges
- Fruits tropicaux
- Herbes et épices
- Notes gourmandes
- Notes salées

RÈGLES :
1. Analyse toute la conversation.
2. Ne redemande jamais une information déjà donnée.
3. Pose une seule question à la fois.
4. Privilégie les questions utiles.
5. Ne suppose jamais les préférences du client.
6. N'invente aucune caractéristique.
7. Ne recommande aucun arôme absent du catalogue.
8. Ne propose pas une famille incompatible avec la demande du client.
9. Ne présente pas une famille comme compatible sans élément dans la demande.
10. Les données techniques et commerciales doivent provenir du catalogue.
11. Si le besoin est suffisamment précis, indique que les candidats peuvent être présentés.
12. Réponds naturellement et en français.

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
        logger.info("[QWEN] Appel du modèle %s", config.LLM_MODEL)

        response = rag.chat(messages)

        logger.info("[QWEN] Réponse reçue.")

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                f"JSON absent. Réponse reçue : {response}"
            )

        result = json.loads(response[start:end + 1])

        status = result.get("status")
        clarification = result.get("question", "")

        if status == "needs_clarification":
            if (
                not isinstance(clarification, str)
                or not clarification.strip()
            ):
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

        raise ValueError(f"Statut invalide : {status}")

    except Exception as exc:
        logger.exception("[ERREUR QWEN] Échec de la génération")

        raise HTTPException(
            status_code=502,
            detail="Le modèle Qwen n'a pas pu traiter la demande.",
        ) from exc


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
        f"Volume minimum : {row['volume_minimum_kg']} kg. "
        f"Source : {row['source_information']}."
    )

    return AromaSource(
        source_type="arome",
        source_id=str(row["aroma_id"]),
        is_synthetic=bool(row["is_synthetic"]),
        text=text,
        score=1.0,
    )


# --------------------------------------------------
# FORMATAGE DES RÉSULTATS
# --------------------------------------------------

def format_catalogue_answer(
    rows: list[dict],
    family: str | None,
    application: str | None,
    numeric_filters: list[dict],
    heat_filter: str | None = None,
) -> str:

    if not rows:
        criteria = []

        if family:
            criteria.append(f"la famille {family}")

        if application:
            criteria.append(f"l'application {application}")

        for criterion in numeric_filters:
            criteria.append(
                f"{criterion['label']} "
                f"{criterion['operator']} "
                f"{criterion['value']}"
            )

        if heat_filter:
            criteria.append(
                f"une tenue à la chaleur {heat_filter}"
            )

        if not criteria:
            return (
                "Je n'ai trouvé aucun candidat correspondant "
                "dans le catalogue."
            )

        return (
            "Aucun arôme ne correspond aux critères suivants : "
            + ", ".join(criteria)
            + "."
        )

    lines = [
        "Voici les arômes correspondant aux critères "
        "identifiés dans le catalogue :"
    ]

    if numeric_filters:
        lines.extend([
            "",
            "**Critères numériques appliqués :**",
        ])

        for criterion in numeric_filters:
            lines.append(
                f"- {criterion['label']} "
                f"{criterion['operator']} "
                f"{criterion['value']}"
            )

    if heat_filter:
        lines.extend([
            "",
            f"**Tenue à la chaleur exigée : {heat_filter}.**",
        ])

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
            f"- Source : {row['source_information']}",
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
        logger.exception("Impossible de vérifier ChromaDB")
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
            query = "SELECT * FROM public.food_aromas WHERE 1=1"
            params = []

            if famille:
                query += " AND famille ILIKE %s"
                params.append(f"%{famille}%")

            if application:
                query += """
                    AND EXISTS (
                        SELECT 1
                        FROM unnest(applications) AS app
                        WHERE app ILIKE %s
                    )
                """
                params.append(f"%{application}%")

            query += " ORDER BY nom"

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    except Exception as exc:
        logger.exception("Impossible de lire le catalogue")

        raise HTTPException(
            status_code=503,
            detail="Impossible de lire le catalogue.",
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
        logger.exception("Erreur pendant la recherche ChromaDB")

        raise HTTPException(
            status_code=503,
            detail="Erreur lors de la recherche dans ChromaDB.",
        ) from exc


@app.post("/ask", tags=["Assistant conversationnel"])
def ask(req: SearchRequest):
    conversation = get_conversation_text(
        req.question,
        req.history,
    )

    rows, family, application, numeric_filters = search_catalogue(
        conversation,
        req.top_k,
    )

    heat_filter = detect_heat_filter(conversation)

    # CAS 1 : critère numérique ou technique détecté.
    if numeric_filters or heat_filter:
        sources = [
            build_source(row)
            for row in rows
        ]

        answer = format_catalogue_answer(
            rows,
            family,
            application,
            numeric_filters,
            heat_filter,
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

    # CAS 2 : aucune préférence sensorielle détectée.
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

    # CAS 3 : appel à Qwen pour guider la conversation.
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

    # CAS 4 : présentation des résultats du catalogue.
    sources = [
        build_source(row)
        for row in rows
    ]

    answer = format_catalogue_answer(
        rows,
        family,
        application,
        numeric_filters,
        heat_filter,
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


# --------------------------------------------------
# LANCEMENT
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
