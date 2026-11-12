"""API Umi : assistant aromatique conversationnel."""

import logging
import re
import unicodedata
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

import config
import rag
from database import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("umi")

MAX_RECOMMENDATIONS = 3

app = FastAPI(
    title="Umi API",
    description="Assistant aromatique conversationnel avec RAG.",
    version="9.1.0",
)


# ==================================================
# MODÈLES
# ==================================================

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(config.TOP_K, ge=1, le=20)
    history: list[dict] = Field(default_factory=list)


class AromaSource(BaseModel):
    source_type: str
    source_id: str
    is_synthetic: bool
    text: str
    score: float = 1.0

    # Champs structurés pour les cartes du frontend
    nom: str | None = None
    famille: str | None = None
    notes: str | None = None
    applications: list[str] = Field(default_factory=list)
    sucrosite: float | None = None
    tenue_chaleur: str | None = None
    prix: float | None = None
    delai: int | None = None


# ==================================================
# NORMALISATION
# ==================================================

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_sentence(value) -> str:
    if value is None:
        return ""
    return str(value).strip().rstrip(" .;") + "."


def get_conversation_text(question, history):
    messages = []

    for item in history[-20:]:
        if (
            isinstance(item, dict)
            and item.get("role") == "user"
            and isinstance(item.get("content"), str)
        ):
            messages.append(item["content"])

    messages.append(question)
    return " ".join(messages)


def get_last_assistant_message(history):
    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ):
            return item["content"]

    return ""


# ==================================================
# DÉMARRAGE
# ==================================================

@app.get("/start", tags=["Assistant conversationnel"])
def start_conversation():
    return {
        "status": "conversation_started",
        "answer": (
            "Bonjour ! Bienvenue chez Umi. "
            "Je vais t'aider à trouver l'arôme qui correspond à ton besoin.\n\n"
            "**Pour commencer, quel type de produit souhaites-tu aromatiser ?**\n\n"
            "Par exemple : une boisson, une pâtisserie, du chocolat, "
            "une confiserie, un produit laitier ou un autre produit."
        ),
        "model": config.LLM_MODEL,
        "sources": [],
        "suggestions": [
            "Boissons",
            "Pâtisserie",
            "Chocolat",
            "Confiserie",
            "Produits laitiers",
        ],
    }


# ==================================================
# ACCÈS À POSTGRESQL
# ==================================================

def get_aroma_by_name(name: str):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM public.food_aromas
                WHERE nom ILIKE %s
                LIMIT 1
                """,
                (name.strip(),),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    finally:
        if conn:
            conn.close()


def get_base_aroma_name(name: str) -> str:
    return name.split(" - ")[0].strip()


def get_aroma_from_user_text(question, history):
    """Retrouve un arôme cité dans la dernière liste proposée."""

    recommendations = get_previous_recommendations(history)
    q = normalize(question)

    if not recommendations:
        return None

    number_match = re.search(
        r"\b(?:la|le|numero|num|n°|#)?\s*([1-3])\b",
        q,
    )

    if number_match:
        index = int(number_match.group(1)) - 1
        if index < len(recommendations):
            return recommendations[index]

    ordinals = {
        "premiere": 0,
        "premier": 0,
        "deuxieme": 1,
        "second": 1,
        "seconde": 1,
        "troisieme": 2,
    }

    for word, index in ordinals.items():
        if re.search(r"\b" + word + r"\b", q):
            if index < len(recommendations):
                return recommendations[index]

    for name in recommendations:
        base = get_base_aroma_name(name)

        if normalize(base) in q or normalize(name) in q:
            return name

    return None


def get_previous_recommendations(history):
    """Récupère la dernière liste de recommandations."""

    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ):
            names = re.findall(
                r"\*\*\s*\d+\.\s*(.+?)\*\*",
                item["content"],
            )

            if names:
                return names[:MAX_RECOMMENDATIONS]

    return []


def get_last_selected_aroma(history):
    """Retrouve le dernier choix individuel de l'utilisateur."""

    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ):
            content = item["content"]

            if (
                "Repartons sur une nouvelle recherche" in content
                or "Quel type de produit souhaites-tu aromatiser ?" in content
            ):
                return None

            match = re.search(
                r"Tu as sélectionné\s*\*\*(.+?)\*\*",
                content,
                re.IGNORECASE,
            )

            if match:
                return match.group(1)

    return None


def get_last_blend(history):
    """Retrouve les deux arômes d'un mélange proposé ou retenu."""

    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ):
            content = item["content"]

            if "Repartons sur une nouvelle recherche" in content:
                return None

            match = re.search(
                r"(?:Mélange proposé|Mélange suivi|Mélange retenu)\s*:\s*"
                r"\*\*(.+?)\s*\+\s*(.+?)\*\*",
                content,
                re.IGNORECASE,
            )

            if match:
                return match.group(1).strip(), match.group(2).strip()

    return None


# ==================================================
# DÉTECTION DES INTENTIONS
# ==================================================

def is_new_search_request(question):
    q = normalize(question)

    return any(x in q for x in [
        "nouvelle recherche",
        "nouvel arome",
        "recommencer",
        "repartir de zero",
        "autre recherche",
        "chercher un autre arome",
        "changer d arome",
        "oublie le choix",
        "on recommence",
    ])


def is_comparison_request(question):
    q = normalize(question)
    return any(x in q for x in [
        "comparer", "comparaison", "compare",
    ])


def is_more_info_request(question):
    q = normalize(question)
    return any(x in q for x in [
        "en savoir plus", "plus d information", "plus d info",
        "plus de detail", "detaille", "explique",
        "approfondir", "parle moi de", "parles moi de",
    ])


def is_commercial_request(question):
    q = normalize(question)
    return any(x in q for x in [
        "prix", "cout", "tarif", "delai",
        "volume minimum", "quantite minimale",
        "conditions commerciales",
    ])


def is_purchase_confirmation(question):
    q = normalize(question)
    return any(x in q for x in [
        "je prends", "je vais prendre", "je choisis",
        "je choisi", "je valide", "je retiens",
        "je pars sur", "c est decide",
        "je veux celui la", "je veux celle la",
    ])


def is_yes(question):
    q = normalize(question)
    return q in [
        "oui", "oui merci", "d accord", "ok",
        "pourquoi pas", "bien sur", "absolument",
        "je veux bien", "avec plaisir",
    ]


def is_no(question):
    q = normalize(question)
    return q in [
        "non", "non merci", "pas de melange",
        "un seul arome", "je prefere un seul arome",
        "je ne veux pas de melange",
        "sans melange", "pas maintenant",
    ]


def asks_another_blend(question):
    q = normalize(question)
    return any(x in q for x in [
        "autre melange", "un autre melange",
        "autre combinaison", "autre association",
        "propose autre chose",
    ])


def has_other_application_answer(question):
    return normalize(question) in [
        "autre", "autre chose", "autres",
        "une autre", "autre application",
    ]


# ==================================================
# FAMILLES ET APPLICATIONS
# ==================================================

def detect_family(question):
    q = normalize(question)

    families = {
        "fruits rouges": "Fruits rouges",
        "fruit rouge": "Fruits rouges",
        "fraise": "Fruits rouges",
        "framboise": "Fruits rouges",
        "cerise": "Fruits rouges",
        "fruits tropicaux": "Fruits tropicaux",
        "fruit exotique": "Fruits tropicaux",
        "fruits exotiques": "Fruits tropicaux",
        "mangue": "Fruits tropicaux",
        "ananas": "Fruits tropicaux",
        "passion": "Fruits tropicaux",
        "agrumes": "Agrumes",
        "agrume": "Agrumes",
        "citron": "Agrumes",
        "orange": "Agrumes",
        "pamplemousse": "Agrumes",
        "mandarine": "Agrumes",
        "vanille": "Notes gourmandes",
        "caramel": "Notes gourmandes",
        "chocolat": "Notes gourmandes",
        "gourmand": "Notes gourmandes",
        "epice": "Herbes et épices",
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


def detect_application(question):
    q = normalize(question)

    applications = {
        "patisserie": "pâtisserie",
        "gateau": "pâtisserie",
        "gateaux": "pâtisserie",
        "biscuit": "biscuits",
        "boisson": "boissons",
        "chocolat": "chocolat",
        "confiserie": "confiserie",
        "dessert": "desserts",
        "glace": "glaces",
        "produit laitier": "produits laitiers",
        "yaourt": "produits laitiers",
        "plat prepare": "plats préparés",
        "sauce": "sauces",
        "soupe": "soupes",
        "snack": "snacks",
    }

    for keyword in sorted(applications, key=len, reverse=True):
        if keyword in q:
            return applications[keyword]

    return None


# ==================================================
# FILTRES
# ==================================================

def detect_sweetness_filter(question):
    q = normalize(question)

    if any(x in q for x in [
        "sans sucre", "tres peu sucre", "sucrosite minimale",
    ]):
        return [{
            "column": "sucrosite",
            "operator": "<=",
            "value": Decimal("1"),
        }]

    if any(x in q for x in [
        "equilibre", "equilibree", "modere",
        "moderee", "moyennement sucre",
    ]):
        return [{
            "column": "sucrosite",
            "operator": "=",
            "value": Decimal("3"),
        }]

    if any(x in q for x in [
        "peu sucre", "pas trop sucre", "moins sucre",
    ]):
        return [{
            "column": "sucrosite",
            "operator": "<=",
            "value": Decimal("3"),
        }]

    if any(x in q for x in [
        "extremement sucre", "sucrosite maximale",
    ]):
        return [{
            "column": "sucrosite",
            "operator": "=",
            "value": Decimal("5"),
        }]

    if any(x in q for x in [
        "tres sucre", "bien sucre", "fortement sucre",
        "beaucoup de sucre", "plutot sucre",
    ]):
        return [{
            "column": "sucrosite",
            "operator": ">=",
            "value": Decimal("4"),
        }]

    return []


def detect_heat_filter(question, application=None):
    q = normalize(question)

    if any(x in q for x in [
        "pasteurise", "pasteurisation", "cuisson",
        "haute temperature", "haute chaleur",
        "resistant a la chaleur", "tenue a la chaleur",
    ]):
        return "élevée"

    if (
        re.search(r"\b(oui|absolument|bien sur)\b", q)
        and application in ("pâtisserie", "biscuits")
    ):
        return "élevée"

    return None


# ==================================================
# RECHERCHE DU CATALOGUE
# ==================================================

def search_catalogue(question, top_k):
    family = detect_family(question)
    application = detect_application(question)
    numeric_filters = detect_sweetness_filter(question)
    heat_filter = detect_heat_filter(question, application)

    if not any([family, application, numeric_filters, heat_filter]):
        return [], family, application, numeric_filters, heat_filter, []

    allowed_columns = {"sucrosite"}
    allowed_operators = {">", "<", ">=", "<=", "="}

    def execute_search(use_family=True, use_application=True):
        conn = None

        try:
            conn = get_connection()

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = "SELECT * FROM public.food_aromas WHERE 1=1"
                params = []

                if family and use_family:
                    query += " AND famille ILIKE %s"
                    params.append(family)

                if application and use_application:
                    query += """
                        AND EXISTS (
                            SELECT 1
                            FROM unnest(applications) AS app
                            WHERE app ILIKE %s
                        )
                    """
                    params.append(f"%{application}%")

                for criterion in numeric_filters:
                    column = criterion["column"]
                    operator = criterion["operator"]

                    if (
                        column in allowed_columns
                        and operator in allowed_operators
                    ):
                        query += f" AND {column} {operator} %s"
                        params.append(criterion["value"])

                if heat_filter:
                    query += " AND tenue_chaleur ILIKE %s"
                    params.append(f"%{heat_filter}%")

                if numeric_filters:
                    query += " ORDER BY sucrosite DESC NULLS LAST, nom ASC"
                else:
                    query += " ORDER BY nom ASC"

                query += " LIMIT %s"
                params.append(top_k)

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        finally:
            if conn:
                conn.close()

    try:
        rows = execute_search(True, True)
        if rows:
            return rows, family, application, numeric_filters, heat_filter, []

        if application:
            rows = execute_search(True, False)
            if rows:
                return (
                    rows, family, application, numeric_filters,
                    heat_filter, ["application"],
                )

        if family:
            rows = execute_search(False, False)
            if rows:
                return (
                    rows, family, application, numeric_filters,
                    heat_filter, ["famille", "application"],
                )

        return [], family, application, numeric_filters, heat_filter, []

    except Exception as exc:
        logger.exception("Erreur PostgreSQL")
        raise HTTPException(
            status_code=503,
            detail="Erreur lors de la recherche dans PostgreSQL.",
        ) from exc


# ==================================================
# PRÉSENTATION DES ARÔMES
# ==================================================

def format_selected_aroma(row):
    lines = [
        f"Excellent choix ! Tu as sélectionné **{row['nom']}**.",
        "",
        f"Famille aromatique : {row.get('famille') or 'Non renseignée'}.",
    ]

    if row.get("notes"):
        lines.append(f"Profil : {clean_sentence(row['notes'])}")

    if row.get("applications"):
        lines.append(
            "Applications : " + ", ".join(row["applications"]) + "."
        )

    if row.get("sucrosite") is not None:
        lines.append(f"Sucrosité : {row['sucrosite']}/5.")

    if row.get("tenue_chaleur"):
        lines.append(f"Tenue à la chaleur : {row['tenue_chaleur']}.")

    if row.get("prix_indicatif_eur_kg") is not None:
        lines.append(
            f"Prix indicatif : {row['prix_indicatif_eur_kg']} €/kg."
        )

    if row.get("delai_jours") is not None:
        lines.append(f"Délai indicatif : {row['delai_jours']} jours.")

    if row.get("volume_minimum_kg") is not None:
        lines.append(
            f"Volume minimum : {row['volume_minimum_kg']} kg."
        )

    lines.extend([
        "",
        "Je garde cet arôme comme ton choix actuel.",
        "",
        "Tu peux maintenant demander des détails, consulter les "
        "informations commerciales, le comparer ou rechercher un nouvel arôme.",
    ])

    return "\n".join(lines)


def format_more_info(row):
    lines = [
        f"Bien sûr ! Approfondissons **{row['nom']}**.",
        "",
    ]

    if row.get("notes"):
        lines.append(f"**Profil aromatique :** {clean_sentence(row['notes'])}")

    if row.get("famille"):
        lines.append(f"**Famille :** {row['famille']}.")

    if row.get("applications"):
        lines.append(
            "**Applications :** " + ", ".join(row["applications"]) + "."
        )

    if row.get("certifications"):
        lines.append(
            "**Certifications :** " + ", ".join(row["certifications"]) + "."
        )
    else:
        lines.append(
            "**Certifications :** aucune information renseignée "
            "dans le catalogue."
        )

    if row.get("source_information"):
        lines.append(
            f"**Source :** {row['source_information']}."
        )

    lines.extend([
        "",
        f"Arôme suivi : **{row['nom']}**.",
        "Souhaites-tu consulter son prix, le comparer ou découvrir "
        "une association avec un autre arôme ?",
    ])

    return "\n".join(lines)


def format_commercial_info(row):
    lines = [
        f"Informations commerciales pour **{row['nom']}** :",
        "",
    ]

    found = False

    fields = [
        ("prix_indicatif_eur_kg", "Prix indicatif", " €/kg"),
        ("delai_jours", "Délai indicatif", " jours"),
        ("volume_minimum_kg", "Volume minimum", " kg"),
    ]

    for key, label, suffix in fields:
        if row.get(key) is not None:
            lines.append(f"- {label} : {row[key]}{suffix}.")
            found = True

    if not found:
        lines.append("Aucune information commerciale n'est renseignée.")

    lines.extend([
        "",
        "Souhaites-tu comparer ces informations avec un autre arôme ?",
    ])

    return "\n".join(lines)


# ==================================================
# COMPARAISON DE DEUX ARÔMES
# ==================================================

def format_comparison(first, second):
    lines = [
        f"Voici la comparaison entre **{first['nom']}** "
        f"et **{second['nom']}** :",
        "",
    ]

    fields = [
        ("famille", "Famille"),
        ("sucrosite", "Sucrosité"),
        ("tenue_chaleur", "Tenue à la chaleur"),
        ("prix_indicatif_eur_kg", "Prix indicatif (€/kg)"),
        ("delai_jours", "Délai (jours)"),
        ("volume_minimum_kg", "Volume minimum (kg)"),
    ]

    for key, label in fields:
        value_a = first.get(key)
        value_b = second.get(key)

        if value_a is None and value_b is None:
            continue

        if key == "sucrosite":
            if value_a is not None:
                value_a = f"{value_a}/5"
            if value_b is not None:
                value_b = f"{value_b}/5"

        lines.append(
            f"- **{label} :** "
            f"{value_a if value_a is not None else 'Non renseigné'} "
            f"contre "
            f"{value_b if value_b is not None else 'Non renseigné'}."
        )

    lines.extend([
        "",
        "Les données absentes ne sont pas estimées.",
        "",
        f"Ton choix initial reste **{first['nom']}**.",
        f"Souhaites-tu conserver cet arôme ou choisir **{second['nom']}** ?",
    ])

    return "\n".join(lines)


# ==================================================
# MÉLANGES AROMATIQUES
# ==================================================

def can_suggest_blend(first, second):
    if not first or not second:
        return False

    if first.get("aroma_id") == second.get("aroma_id"):
        return False

    family_a = normalize(first.get("famille") or "")
    family_b = normalize(second.get("famille") or "")

    applications_a = first.get("applications") or []
    applications_b = second.get("applications") or []

    common_applications = {
        normalize(x) for x in applications_a
    } & {
        normalize(x) for x in applications_b
    }

    return bool(
        family_a
        and family_b
        and family_a == family_b
        and common_applications
    )


def format_blend_offer(first, second):
    return (
        "J'ai aussi une idée d'association aromatique !\n\n"
        f"**{first['nom']} + {second['nom']}**\n\n"
        f"Les deux références appartiennent à la famille "
        f"{first.get('famille') or 'non renseignée'} "
        "et partagent au moins une application au catalogue.\n\n"
        "Cela peut être une piste intéressante à explorer. "
        "Attention : cette association n'est pas une compatibilité "
        "technique validée.\n\n"
        "**Souhaites-tu essayer ce mélange ?**\n\n"
        "- Oui, propose-moi ce mélange.\n"
        "- Non, je préfère un seul arôme.\n"
        "- Propose-moi une autre association."
    )


def format_blend_details(first, second):
    return (
        "Voici ta proposition de mélange :\n\n"
        f"**Mélange proposé : {first['nom']} + {second['nom']}**\n\n"
        f"**Premier arôme : {first['nom']}**\n"
        f"- Famille : {first.get('famille') or 'Non renseignée'}.\n"
        f"- Profil : {clean_sentence(first.get('notes')) if first.get('notes') else 'Non renseigné'}\n\n"
        f"**Deuxième arôme : {second['nom']}**\n"
        f"- Famille : {second.get('famille') or 'Non renseignée'}.\n"
        f"- Profil : {clean_sentence(second.get('notes')) if second.get('notes') else 'Non renseigné'}\n\n"
        "Aucun dosage n'est validé par le catalogue. "
        "Les proportions devront être déterminées par des essais "
        "de formulation, après vérification des contraintes techniques "
        "et réglementaires.\n\n"
        f"**Mélange suivi : {first['nom']} + {second['nom']}**\n\n"
        "Souhaites-tu conserver ce mélange ou revenir à un seul arôme ?"
    )


def build_blend_sources(first, second):
    return [
        build_source(first).model_dump(),
        build_source(second).model_dump(),
    ]


# ==================================================
# SOURCES
# ==================================================

def build_source(row):
    applications = row.get("applications") or []

    fields = [
        f"Arôme : {row.get('nom')}",
        f"Famille : {row.get('famille')}",
        f"Notes : {row.get('notes')}",
        f"Applications : {applications}",
        f"Sucrosité : {row.get('sucrosite')}",
        f"Tenue à la chaleur : {row.get('tenue_chaleur')}",
        f"Certifications : {row.get('certifications')}",
        f"Prix : {row.get('prix_indicatif_eur_kg')}",
        f"Délai : {row.get('delai_jours')}",
        f"Volume minimum : {row.get('volume_minimum_kg')}",
        f"Source : {row.get('source_information')}",
    ]

    return AromaSource(
        source_type="arome",
        source_id=str(row.get("aroma_id", "")),
        is_synthetic=bool(row.get("is_synthetic")),
        text=". ".join(str(x) for x in fields) + ".",
        score=1.0,
        nom=row.get("nom"),
        famille=row.get("famille"),
        notes=row.get("notes"),
        applications=applications,
        sucrosite=(
            float(row["sucrosite"])
            if row.get("sucrosite") is not None
            else None
        ),
        tenue_chaleur=row.get("tenue_chaleur"),
        prix=(
            float(row["prix_indicatif_eur_kg"])
            if row.get("prix_indicatif_eur_kg") is not None
            else None
        ),
        delai=(
            int(row["delai_jours"])
            if row.get("delai_jours") is not None
            else None
        ),
    )


# ==================================================
# QUESTIONS PROGRESSIVES
# ==================================================

def get_next_question(conversation, family, application, filters, heat):
    sweetness_known = any(
        x["column"] == "sucrosite" for x in filters
    )

    if not application:
        return (
            "Pour quel type de produit souhaites-tu utiliser ton arôme ? "
            "Par exemple : boisson, pâtisserie, chocolat, confiserie "
            "ou produit laitier."
        )

    if not family:
        return (
            "Quel univers aromatique t'intéresse le plus : "
            "fruité, gourmand, agrumes, épicé ou salé ?"
        )

    if not sweetness_known:
        return (
            "Concernant la sucrosité, tu préfères un arôme peu sucré, "
            "équilibré ou très sucré ?"
        )

    if application in ("pâtisserie", "biscuits") and not heat:
        return (
            "Ton arôme doit-il résister à la cuisson "
            "ou à une température élevée ?"
        )

    return (
        "As-tu une autre préférence, par exemple sur l'intensité, "
        "le prix ou la tenue à la chaleur ?"
    )


def count_known_criteria(family, application, filters, heat):
    count = sum(bool(x) for x in [family, application, heat])

    if any(x["column"] == "sucrosite" for x in filters):
        count += 1

    return count


# ==================================================
# RECOMMANDATIONS
# ==================================================

def format_recommendations(rows, relaxed=None):
    if not rows:
        return (
            "Je n'ai pas trouvé d'arôme correspondant à tes critères.\n\n"
            "Souhaites-tu assouplir certains critères pour élargir "
            "la recherche ?"
        )

    selected = rows[:MAX_RECOMMENDATIONS]
    lines = ["J'ai trouvé quelques pistes intéressantes pour toi !", ""]

    if relaxed:
        lines.extend([
            "J'ai élargi la recherche sur : "
            + ", ".join(relaxed) + ".",
            "",
        ])

    for index, row in enumerate(selected, start=1):
        lines.append(f"**{index}. {row['nom']}**")
        lines.append(f"Famille : {row.get('famille') or 'Non renseignée'}.")

        if row.get("notes"):
            lines.append(f"Profil : {clean_sentence(row['notes'])}")

        if row.get("sucrosite") is not None:
            lines.append(f"Sucrosité : {row['sucrosite']}/5.")

        if row.get("tenue_chaleur"):
            lines.append(f"Tenue à la chaleur : {row['tenue_chaleur']}.")

        lines.append("")

    lines.extend([
        "Ces propositions proviennent de ton catalogue aromatique.",
        "",
        "Laquelle de ces pistes te plaît le plus ? "
        "Tu peux répondre par son numéro ou son nom.",
    ])

    if len(selected) >= 2 and can_suggest_blend(selected[0], selected[1]):
        lines.extend([
            "",
            f"**Une autre idée :** associer {selected[0]['nom']} "
            f"et {selected[1]['nom']}.",
            "Souhaites-tu également explorer ce mélange ? "
            "Tu peux répondre oui ou non.",
        ])

    return "\n".join(lines)


# ==================================================
# ROUTES SYSTÈME ET CATALOGUE
# ==================================================

@app.get("/health", tags=["Système"])
def health():
    try:
        count = rag.get_collection().count()
    except Exception:
        logger.exception("Impossible de vérifier ChromaDB")
        count = None

    return {
        "status": "online",
        "service": "Umi",
        "documents_indexed": count,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/aromes", tags=["Catalogue aromatique"])
def get_aromes(famille: str | None = None, application: str | None = None):
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
        if conn:
            conn.close()


@app.post("/search", response_model=list[AromaSource], tags=["RAG aromatique"])
def search(req: SearchRequest):
    try:
        documents = rag.search_documents(req.question, req.top_k)

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


# ==================================================
# ROUTE PRINCIPALE DE CONVERSATION
# ==================================================

@app.post("/ask", tags=["Assistant conversationnel"])
def ask(req: SearchRequest):
    question = req.question.strip()
    history = req.history
    last_message = get_last_assistant_message(history)

    def response(status, answer, sources=None, suggestions=None):
        return {
            "question": question,
            "status": status,
            "answer": answer,
            "model": config.LLM_MODEL,
            "sources": sources or [],
            "suggestions": suggestions or [],
        }

    # ----------------------------------------------
    # 1. NOUVELLE RECHERCHE
    # ----------------------------------------------

    if is_new_search_request(question):
        return response(
            "needs_clarification",
            "Avec plaisir ! Repartons sur une nouvelle recherche.\n\n"
            "**Quel type de produit souhaites-tu aromatiser ?**",
            suggestions=[
                "Boissons",
                "Pâtisserie",
                "Chocolat",
                "Confiserie",
                "Produits laitiers",
            ],
        )

    # ----------------------------------------------
    # 2. REFUS D'UN MÉLANGE
    # ----------------------------------------------

    if (
        is_no(question)
        and (
            "Souhaites-tu essayer ce mélange" in last_message
            or "Souhaites-tu également explorer ce mélange" in last_message
        )
    ):
        return response(
            "blend_declined",
            "Aucun problème ! Nous allons rester sur un seul arôme.\n\n"
            "Tu peux choisir l'une des propositions individuelles "
            "que je t'ai présentées.",
            suggestions=[
                "Choisir le premier arôme",
                "Choisir le deuxième arôme",
                "Choisir le troisième arôme",
            ],
        )

    # ----------------------------------------------
    # 3. DEMANDE D'UN AUTRE MÉLANGE
    # ----------------------------------------------

    if asks_another_blend(question):
        recommendations = get_previous_recommendations(history)

        if len(recommendations) >= 2:
            rows = [
                get_aroma_by_name(name)
                for name in recommendations
            ]
            rows = [row for row in rows if row]

            pairs = [
                (rows[i], rows[j])
                for i in range(len(rows))
                for j in range(i + 1, len(rows))
                if can_suggest_blend(rows[i], rows[j])
            ]

            if pairs:
                first, second = pairs[-1]
                return response(
                    "blend_proposed",
                    format_blend_offer(first, second),
                    build_blend_sources(first, second),
                    suggestions=[
                        "Oui, proposer ce mélange",
                        "Non, garder un seul arôme",
                    ],
                )

        return response(
            "needs_clarification",
            "Je n'ai pas trouvé d'autre association suffisamment "
            "pertinente parmi les propositions actuelles. "
            "Souhaites-tu lancer une nouvelle recherche ?",
            suggestions=["Nouvelle recherche"],
        )

    # ----------------------------------------------
    # 4. ACCEPTATION D'UN MÉLANGE
    # ----------------------------------------------

    if (
        is_yes(question)
        and (
            "Souhaites-tu essayer ce mélange" in last_message
            or "Souhaites-tu également explorer ce mélange" in last_message
        )
    ):
        recommendations = get_previous_recommendations(history)

        if len(recommendations) >= 2:
            rows = [
                get_aroma_by_name(name)
                for name in recommendations[:2]
            ]

            if all(rows) and can_suggest_blend(rows[0], rows[1]):
                return response(
                    "blend_selected",
                    format_blend_details(rows[0], rows[1]),
                    build_blend_sources(rows[0], rows[1]),
                    suggestions=[
                        "Conserver ce mélange",
                        "Revenir à un seul arôme",
                        "Nouvelle recherche",
                    ],
                )

        return response(
            "needs_clarification",
            "Je n'arrive plus à retrouver les deux références du "
            "mélange. Peux-tu me préciser les deux arômes souhaités ?",
        )

    # ----------------------------------------------
    # 5. SUIVI D'UN MÉLANGE DÉJÀ PROPOSÉ
    # ----------------------------------------------

    previous_blend = get_last_blend(history)

    if previous_blend:
        first_name, second_name = previous_blend

        try:
            first = get_aroma_by_name(first_name)
            second = get_aroma_by_name(second_name)

            if first and second:
                if is_comparison_request(question):
                    return response(
                        "comparison_requested",
                        f"Ton mélange actuel est **{first['nom']} + "
                        f"{second['nom']}**. Quel autre arôme souhaites-tu "
                        "comparer à ce mélange ?",
                        build_blend_sources(first, second),
                        suggestions=[
                            "Comparer avec un autre arôme",
                            "Nouvelle recherche",
                        ],
                    )

                if is_more_info_request(question):
                    return response(
                        "blend_details",
                        format_blend_details(first, second),
                        build_blend_sources(first, second),
                    )

                if is_commercial_request(question):
                    return response(
                        "blend_commercial_details",
                        "Voici les informations commerciales disponibles "
                        "pour les deux références :\n\n"
                        + format_comparison(first, second),
                        build_blend_sources(first, second),
                    )

                if is_purchase_confirmation(question):
                    return response(
                        "blend_confirmed",
                        f"Parfait ! Je conserve le mélange "
                        f"**{first['nom']} + {second['nom']}** "
                        "comme ton choix actuel.\n\n"
                        "Les proportions devront être validées par "
                        "des essais avant toute production.",
                        build_blend_sources(first, second),
                    )

        except Exception as exc:
            logger.exception("Erreur lors du suivi du mélange")
            raise HTTPException(
                status_code=503,
                detail="Impossible de retrouver le mélange.",
            ) from exc

    # ----------------------------------------------
    # 6. CHOIX D'UN SECOND ARÔME POUR COMPARAISON
    # ----------------------------------------------

    previous_selection = get_last_selected_aroma(history)

    comparison_pending = (
        "Quel arôme souhaites-tu lui comparer" in last_message
    )

    if comparison_pending and previous_selection:
        comparison_name = get_aroma_from_user_text(question, history)

        if comparison_name:
            try:
                first = get_aroma_by_name(previous_selection)
                second = get_aroma_by_name(comparison_name)

                if first and second:
                    return response(
                        "comparison_ready",
                        format_comparison(first, second),
                        [
                            build_source(first).model_dump(),
                            build_source(second).model_dump(),
                        ],
                        suggestions=[
                            "Conserver le premier arôme",
                            "Choisir le deuxième arôme",
                            "Explorer un mélange",
                        ],
                    )

            except Exception as exc:
                logger.exception("Erreur pendant la comparaison")
                raise HTTPException(
                    status_code=503,
                    detail="Impossible de comparer les arômes.",
                ) from exc

        return response(
            "needs_clarification",
            "Je n'ai pas retrouvé cet arôme dans les propositions. "
            "Peux-tu me donner son nom exact ?",
        )

    # ----------------------------------------------
    # 7. CHOIX D'UN ARÔME APRÈS UNE COMPARAISON
    # ----------------------------------------------

    if "Souhaites-tu conserver cet arôme" in last_message:
        selected_name = get_aroma_from_user_text(question, history)

        if selected_name:
            row = get_aroma_by_name(selected_name)

            if row:
                return response(
                    "aroma_selected",
                    format_selected_aroma(row),
                    [build_source(row).model_dump()],
                    suggestions=[
                        "En savoir plus",
                        "Informations commerciales",
                        "Comparer",
                        "Nouvelle recherche",
                    ],
                )

        if is_purchase_confirmation(question):
            return response(
                "aroma_confirmed",
                "Parfait ! Je conserve ton choix actuel.",
            )

    # ----------------------------------------------
    # 8. SUIVI D'UN ARÔME SÉLECTIONNÉ
    # ----------------------------------------------

    if previous_selection:
        try:
            selected_row = get_aroma_by_name(previous_selection)

            if selected_row:
                if is_comparison_request(question):
                    return response(
                        "comparison_requested",
                        f"Très bien ! Nous allons comparer "
                        f"**{selected_row['nom']}** avec un autre arôme.\n\n"
                        "Quel arôme souhaites-tu lui comparer ? "
                        "Tu peux me donner son nom.",
                        [build_source(selected_row).model_dump()],
                        suggestions=[
                            "Choisir un autre arôme",
                            "Nouvelle recherche",
                        ],
                    )

                if is_commercial_request(question):
                    return response(
                        "commercial_details",
                        format_commercial_info(selected_row),
                        [build_source(selected_row).model_dump()],
                    )

                if is_more_info_request(question):
                    return response(
                        "aroma_details",
                        format_more_info(selected_row),
                        [build_source(selected_row).model_dump()],
                    )

                alternative_name = get_aroma_from_user_text(
                    question, history
                )

                if alternative_name:
                    new_row = get_aroma_by_name(alternative_name)

                    if new_row:
                        return response(
                            "aroma_selected",
                            format_selected_aroma(new_row),
                            [build_source(new_row).model_dump()],
                        )

                if is_purchase_confirmation(question):
                    return response(
                        "aroma_confirmed",
                        f"Parfait ! Je conserve **{selected_row['nom']}** "
                        "comme ton choix actuel.\n\n"
                        "Tu peux demander un devis ou consulter "
                        "une dernière information.",
                        [build_source(selected_row).model_dump()],
                    )

        except Exception as exc:
            logger.exception("Erreur pendant le suivi de l'arôme")
            raise HTTPException(
                status_code=503,
                detail="Impossible de récupérer l'arôme sélectionné.",
            ) from exc

    # ----------------------------------------------
    # 9. PREMIÈRE SÉLECTION DANS LES RECOMMANDATIONS
    # ----------------------------------------------

    selected_name = get_aroma_from_user_text(question, history)

    if selected_name:
        try:
            selected_row = get_aroma_by_name(selected_name)

            if selected_row:
                return response(
                    "aroma_selected",
                    format_selected_aroma(selected_row),
                    [build_source(selected_row).model_dump()],
                    suggestions=[
                        "En savoir plus",
                        "Informations commerciales",
                        "Comparer",
                        "Nouvelle recherche",
                    ],
                )

        except Exception as exc:
            logger.exception("Erreur lors de la sélection")
            raise HTTPException(
                status_code=503,
                detail="Impossible de récupérer cet arôme.",
            ) from exc

    # ----------------------------------------------
    # 10. DEMANDE DE COMPARAISON SANS CHOIX ACTUEL
    # ----------------------------------------------

    if is_comparison_request(question):
        return response(
            "needs_clarification",
            "Pour comparer deux arômes, sélectionne d'abord "
            "une proposition. Nous pourrons ensuite choisir "
            "la deuxième référence.",
            suggestions=[
                "Choisir le premier arôme",
                "Choisir le deuxième arôme",
            ],
        )

    # ----------------------------------------------
    # 11. RECHERCHE CONVERSATIONNELLE
    # ----------------------------------------------

    conversation = get_conversation_text(question, history)

    if has_other_application_answer(question):
        return response(
            "needs_clarification",
            "Pas de problème ! Peux-tu préciser ton utilisation ? "
            "Par exemple : sauces, snacks, soupes ou plats préparés.",
            suggestions=[
                "Sauces",
                "Snacks",
                "Soupes",
                "Plats préparés",
            ],
        )

    (
        rows,
        family,
        application,
        numeric_filters,
        heat_filter,
        relaxed,
    ) = search_catalogue(
        conversation,
        max(req.top_k, MAX_RECOMMENDATIONS),
    )

    known_criteria = count_known_criteria(
        family,
        application,
        numeric_filters,
        heat_filter,
    )

    needs_heat_answer = (
        application in ("pâtisserie", "biscuits")
        and not heat_filter
    )

    explicit_request = any(
        x in normalize(question)
        for x in [
            "propose moi", "recommande moi",
            "montre moi", "quels aromes",
            "quelles aromes", "donne moi des propositions",
            "je veux voir les resultats", "c est tout",
            "tu peux proposer",
        ]
    )

    # La famille ET l'application sont obligatoires.
    # Une demande explicite ne peut pas contourner cette règle.
    if not application or not family:
        return response(
            "needs_clarification",
            get_next_question(
                conversation,
                family,
                application,
                numeric_filters,
                heat_filter,
            ),
            suggestions=(
                [
                    "Boissons",
                    "Pâtisserie",
                    "Chocolat",
                    "Confiserie",
                    "Produits laitiers",
                    "Biscuits",
                ]
                if not application
                else [
                    "Fruits rouges",
                    "Fruits tropicaux",
                    "Agrumes",
                    "Notes gourmandes",
                    "Herbes et épices",
                    "Notes salées",
                ]
            ),
        )

    # Les autres critères peuvent être demandés progressivement.
    if (known_criteria < 3 or needs_heat_answer) and not explicit_request:
        return response(
            "needs_clarification",
            get_next_question(
                conversation,
                family,
                application,
                numeric_filters,
                heat_filter,
            ),
            suggestions=(
                [
                    "Peu sucré",
                    "Équilibré",
                    "Très sucré",
                ]
                if not any(
                    x["column"] == "sucrosite"
                    for x in numeric_filters
                )
                else [
                    "Oui, résistance à la cuisson nécessaire",
                    "Non, pas de cuisson",
                ]
            ),
        )

    # ----------------------------------------------
    # 12. RECOMMANDATIONS FINALES
    # ----------------------------------------------

    sources = [
        build_source(row).model_dump()
        for row in rows[:MAX_RECOMMENDATIONS]
    ]

    suggestions = [
        "Choisir le premier arôme",
        "Choisir le deuxième arôme",
        "Comparer deux arômes",
        "Nouvelle recherche",
    ]

    if len(rows) >= 2 and can_suggest_blend(rows[0], rows[1]):
        suggestions.insert(3, "Explorer un mélange")

    return response(
        "recommendations_ready",
        format_recommendations(rows, relaxed),
        sources,
        suggestions=suggestions,
    )


# ==================================================
# LANCEMENT
# ==================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )