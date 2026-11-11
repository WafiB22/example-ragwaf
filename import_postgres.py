
import csv
import psycopg2
from pathlib import Path
from getpass import getpass

CSV_PATH = Path(__file__).parent / "data" / "aromes.csv"

def importer():
    mot_de_passe = getpass("Mot de passe PostgreSQL : ")

    connexion = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="olfactive_studio",
        user="postgres",
        password=mot_de_passe
    )

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as fichier:
        lignes = (ligne for ligne in fichier if ligne.strip())
        lecteur = csv.DictReader(lignes)

        aromes = list(lecteur)

    with connexion:
        with connexion.cursor() as curseur:
            for arome in aromes:
                curseur.execute("""
                    INSERT INTO aromes (
                        id, nom, famille, notes, applications,
                        sucrosite, tenue_chaleur, certifications,
                        prix_indicatif_eur_kg, delai_jours,
                        volume_minimum_kg, is_synthetic
                    )
                    VALUES (
                        %(id)s, %(nom)s, %(famille)s, %(notes)s,
                        %(applications)s, %(sucrosite)s,
                        %(tenue_chaleur)s, %(certifications)s,
                        %(prix_indicatif_eur_kg)s, %(delai_jours)s,
                        %(volume_minimum_kg)s, %(is_synthetic)s
                    )
                    ON CONFLICT (id) DO NOTHING
                """, arome)

    connexion.close()
    print(f"Importation terminée : {len(aromes)} arômes traités.")

if __name__ == "__main__":
    importer()
