
"""Indexe la documentation aromatique dans ChromaDB.

Structure attendue :
data/
    knowledge/
        familles_aromatiques.txt
        notes_sensorielles.txt
        applications.txt

Lancement :
    python ingest.py
"""

import hashlib
import re
from pathlib import Path

import config
import rag


# Dossier contenant les connaissances aromatiques
KNOWLEDGE_DIR = Path(__file__).parent / "data" / "knowledge"

# Taille maximale indicative d'un passage
CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 1000)


def split_into_chunks(text: str) -> list[str]:
    """Découpe un document en passages adaptés à la recherche."""

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = []
    current_size = 0

    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > CHUNK_SIZE:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        current.append(paragraph)
        current_size += len(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def main():
    """Charge et indexe tous les documents aromatiques."""

    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Dossier créé : {KNOWLEDGE_DIR}")
        print("Ajoute des fichiers .txt dans ce dossier puis relance.")
        return

    files = sorted(KNOWLEDGE_DIR.glob("*.txt"))

    if not files:
        print("Aucun document .txt trouvé dans data/knowledge.")
        return

    ids = []
    documents = []
    metadatas = []

    for path in files:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Document vide ignoré : {path.name}")
            continue

        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            unique_text = f"{path.name}-{index}-{chunk}"
            chunk_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(chunk)

            metadatas.append({
                "source": path.name,
                "category": "aromatic_knowledge",
                "chunk_index": index,
            })

        print(f"{path.name} : {len(chunks)} passages")

    if not documents:
        print("Aucun passage à indexer.")
        return

    print(f"\nTotal : {len(documents)} passages")
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
    print(f"Nombre total de passages : {collection.count()}")
    print(f"Emplacement : {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
