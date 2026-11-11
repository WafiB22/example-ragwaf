
from database import get_connection

connexion = get_connection()
curseur = connexion.cursor()

curseur.execute("SELECT * FROM public.aromes LIMIT 0")

print("Colonnes de la table aromes :")
for colonne in curseur.description:
    print("-", colonne[0])

curseur.close()
connexion.close()
