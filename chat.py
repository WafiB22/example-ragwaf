
import requests

API_URL = "http://127.0.0.1:8001"
TOP_K = 3

history = []

print("\n=== Assistant Arômes ===\n")

# Le chatbot commence la conversation spontanément.
try:
    response = requests.get(
        f"{API_URL}/start",
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    first_message = data.get("answer", "Bonjour ! Que souhaites-tu aromatiser ?")

    print(f"Bot : {first_message}\n")

    history.append({
        "role": "assistant",
        "content": first_message,
    })

except requests.exceptions.RequestException as error:
    print("Impossible de démarrer la conversation.")
    print("Vérifie que python api.py est lancé.")
    print(f"Détail : {error}")
    raise SystemExit(1)


print("Tape 'quit' ou 'exit' pour arrêter.\n")

while True:
    question = input("Toi : ").strip()

    if not question:
        continue

    if question.lower() in ("quit", "exit", "quitter"):
        print("\nFin de la conversation.")
        break

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "top_k": TOP_K,
                "history": history,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        answer = (
            data.get("answer")
            or data.get("response")
            or data.get("message")
        )

        if not answer:
            print("\nRéponse API inattendue :", data)
            continue

        print(f"\nBot : {answer}\n")

        history.append({
            "role": "user",
            "content": question,
        })

        history.append({
            "role": "assistant",
            "content": str(answer),
        })

    except requests.exceptions.ConnectionError:
        print("\nErreur : impossible de joindre l'API.")
        print("Vérifie que python api.py tourne dans l'autre terminal.\n")

    except requests.exceptions.Timeout:
        print("\nLe chatbot met trop de temps à répondre. Réessaie.\n")

    except requests.exceptions.RequestException as error:
        print(f"\nErreur HTTP : {error}\n")
