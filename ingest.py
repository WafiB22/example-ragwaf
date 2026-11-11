
"""Indexe le catalogue CSV et les documents métier dans ChromaDB."""

import csv
import re
from pathlib import Path

import config
import rag

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "aromes.csv"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

REQUIRED_COLUMNS = {
    "id", "nom", "famille", "notes", "applications",
    "sucrosite", "tenue_chaleur", "certifications",
    "prix_indicatif_eur_kg", "delai_jours",
    "volume_minimum_kg", "is_synthetic",
}


def build_aroma_document(row: dict) -> str:
    """Construit une description textuelle de chaque arôme."""
    return (
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


def read_csv() -> list[dict]:
    """Lit le catalogue CSV en détectant son séparateur."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Catalogue introuvable : {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ","

        reader = csv.DictReader(file, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError("Impossible de lire l'en-tête du CSV.")

        reader.fieldnames = [
            name.strip().lstrip("\ufeff") for name in reader.fieldnames
        ]

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Colonnes manquantes : {sorted(missing)}. "
                f"Colonnes détectées : {reader.fieldnames}"
            )

        return list(reader)


def split_text(text: str, max_chars: int = 800) -> list[str]:
    """Découpe un document en morceaux."""
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])
            continue

        candidate = f"{current}\n{paragraph}" if current else paragraph

        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def load_aromas():
    """Prépare les documents du catalogue."""
    rows = read_csv()
    documents = []
    ids = []
    metadatas = []

    for row in rows:
        source_id = (row.get("id") or "").strip()

        if not source_id:
            continue

        documents.append(build_aroma_document(row))
        ids.append(source_id)

        is_synthetic = (
            (row.get("is_synthetic") or "").strip().lower()
            in ("true", "1", "yes", "oui")
        )

        metadatas.append({
            "source_type": "arome",
            "source_id": source_id,
            "is_synthetic": is_synthetic,
        })

    return ids, documents, metadatas


def load_knowledge_documents():
    """Prépare tous les fichiers texte du dossier knowledge."""
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Dossier de connaissances introuvable : {KNOWLEDGE_DIR}"
        )

    ids = []
    documents = []
    metadatas = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8-sig").strip()

        if not content:
            print(f"Document vide ignoré : {file_path.name}")
            continue

        chunks = split_text(content)

        for index, chunk in enumerate(chunks):
            source_id = f"knowledge_{file_path.stem}_{index}"

            ids.append(source_id)
            documents.append(chunk)

            metadatas.append({
                "source_type": "document",
                "source_id": source_id,
                "is_synthetic": True,
            })

        print(
            f"Document chargé : {file_path.name} "
            f"({len(chunks)} morceaux)"
        )

    return ids, documents, metadatas


def main():
    aroma_ids, aroma_docs, aroma_meta = load_aromas()
    knowledge_ids, knowledge_docs, knowledge_meta = (
        load_knowledge_documents()
    )

    ids = aroma_ids + knowledge_ids
    documents = aroma_docs + knowledge_docs
    metadatas = aroma_meta + knowledge_meta

    if not documents:
        raise ValueError("Aucun document à indexer.")

    print(f"\nArômes : {len(aroma_docs)}")
    print(f"Documents métier : {len(knowledge_docs)}")
    print(f"Total à indexer : {len(documents)}")
    print(f"Calcul des embeddings avec {config.EMBED_MODEL}...")

    embeddings = rag.embed_documents(documents)
    collection = rag.get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("\nIndexation terminée !")
    print(f"Collection : {config.COLLECTION_NAME}")
    print(f"Nombre total de documents : {collection.count()}")
    print(f"Emplacement : {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
