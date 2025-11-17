import os
from dotenv import load_dotenv

# Importiamo i nostri moduli personalizzati
try:
    from extractor import extract_knowledge
    from graph import GraphTool
    # IMPORTIAMO IL NOSTRO NUOVO SCRAPER
    from scraper import scrape_article_text
except ImportError as e:
    print(
        f"Errore: Impossibile importare i moduli necessari. Assicurati che 'extractor.py', 'graph.py' e 'scraper.py' siano nella stessa cartella.")
    print(f"Dettaglio: {e}")
    exit(1)

# Definiamo l'URL da analizzare
# Questo è l'articolo che hai appena testato con successo
URL_DA_ANALIZZARE = "https://techcrunch.com/2025/11/10/sapphire-sport-spins-out-rebrands-as-359-capital-with-300m-aum/"


def main():
    """
    Esegue l'intera pipeline del Ciclo 1 in modo automatizzato:
    URL -> Scraper -> Estrazione LLM -> Scrittura su Grafo.
    """
    print("--- AVVIO PIPELINE CICLO 1 (Automatizzata) ---")

    # 1. Carica le variabili d'ambiente
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("NEO4J_URI"):
        raise ValueError(
            "Errore: mancano le API key (GOOGLE_API_KEY) o le credenziali Neo4j (NEO4J_URI, etc.) nel file .env")

    # 2. Esegui lo Scraper (Chiama scraper.py)
    # Sostituiamo la lettura da file con la chiamata allo scraper
    try:
        text_to_analyze = scrape_article_text(URL_DA_ANALIZZARE)
        if not text_to_analyze:
            raise Exception(f"Scraping fallito dall'URL: {URL_DA_ANALIZZARE}")

        print(f"\nTesto caricato via web: {len(text_to_analyze)} caratteri.")
        print(f"Anteprima: {text_to_analyze[:100]}...")
    except Exception as e:
        print(f"Errore durante lo scraping: {e}")
        return

    # 3. Inizializza il Grafo
    graph_tool = None
    try:
        graph_tool = GraphTool(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        if not graph_tool.driver:
            raise Exception("Inizializzazione GraphTool fallita.")

        # 4. Esegui l'Estrazione (Chiama extractor.py)
        knowledge_data = extract_knowledge(text_to_analyze)

        if not knowledge_data:
            raise Exception("Estrazione fallita. Controlla l'output di extractor.py.")

        print(f"\nEstrazione riuscita. Trovate {len(knowledge_data.fondazioni)} fondazioni, "
              f"{len(knowledge_data.investimenti)} investimenti, "
              f"{len(knowledge_data.fallimenti)} fallimenti.")

        # 5. Esegui la Scrittura (Chiama graph.py)
        graph_tool.import_extracted_data(knowledge_data)

        print("\n--- PIPELINE COMPLETATA CON SUCCESSO ---")
        print("Il Knowledge Graph è stato popolato con i dati estratti dall'articolo web.")

    except Exception as e:
        print(f"\n--- ❌ ERRORE NELLA PIPELINE ---")
        print(f"Errore: {e}")
    finally:
        # 6. Chiudi la connessione
        if graph_tool:
            graph_tool.close()


if __name__ == "__main__":
    main()