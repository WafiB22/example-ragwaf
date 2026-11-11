
import csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "data" / "aromes.csv"

def lire_aromes():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

    with open(CSV_PATH, mode="r", encoding="utf-8-sig", newline="") as fichier:
        lignes = (ligne for ligne in fichier if ligne.strip())
        lecteur = csv.DictReader(lignes)
        aromes = list(lecteur)

    champs_attendus = {
        "id", "nom", "famille", "notes", "applications",
        "sucrosite", "tenue_chaleur", "certifications",
        "prix_indicatif_eur_kg", "delai_jours",
        "volume_minimum_kg", "is_synthetic"
    }

    if not aromes:
        raise ValueError("Le catalogue est vide.")

    if not champs_attendus.issubset(lecteur.fieldnames or []):
        raise ValueError("Certaines colonnes obligatoires sont absentes.")

    for arome in aromes:
        arome["sucrosite"] = int(arome["sucrosite"])
        arome["prix_indicatif_eur_kg"] = float(arome["prix_indicatif_eur_kg"])
        arome["delai_jours"] = int(arome["delai_jours"])
        arome["volume_minimum_kg"] = float(arome["volume_minimum_kg"])
        arome["is_synthetic"] = arome["is_synthetic"].strip().lower() == "true"

    return aromes


if __name__ == "__main__":
    catalogue = lire_aromes()
    print(f"Catalogue chargé : {len(catalogue)} arômes")
    for arome in catalogue[:5]:
        print(f"- {arome['id']} : {arome['nom']} ({arome['famille']})")
