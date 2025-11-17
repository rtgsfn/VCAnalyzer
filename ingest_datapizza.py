from datapizza.pipeline import IngestionPipeline  # <-- CLASSE CORRETTA
import os

# Definiamo le cartelle
DOCS_DIR = "input_docs"
STORAGE_DIR = "datapizza_storage" # Datapizza salverà qui il suo database

def main():
    """
    Usa Datapizza AI (IngestionPipeline) per creare e salvare un indice RAG
    dai documenti presenti nella cartella 'pitch_decks'.
    """
    print(f"--- Avvio Ingestione Datapizza da '{DOCS_DIR}' ---")

    if not os.path.exists(DOCS_DIR):
        print(f"Errore: La cartella '{DOCS_DIR}' non è stata trovata.")
        print("Azione richiesta: Crea la cartella e aggiungi i pitch deck.")
        return

    # 2. Inizializza la IngestionPipeline di Datapizza
    try:
        # NOTA: Abbiamo rimosso 'embedding_model' da qui.
        # 'persist_path' dice alla pipeline dove salvare l'indice.
        pipeline = IngestionPipeline(
            persist_path=STORAGE_DIR
        )

        # 3. Aggiungi i documenti alla pipeline
        print("Indicizzazione dei documenti in corso (potrebbe richiedere un po' di tempo)...")
        # .run() legge la cartella, fa il chunking e salva
        pipeline.run(data_path=DOCS_DIR)

        print(f"\n--- ✅ Ingestione Completata ---")
        print(f"I documenti sono stati indicizzati e salvati in '{STORAGE_DIR}'.")
        print("Ora puoi eseguire 'analyzer.py'.")

    except Exception as e:
        print(f"\n--- ❌ Errore durante l'ingestione Datapizza ---")
        print(f"Errore: {e}")
        print("Assicurati che 'datapizza-ai' sia installato correttamente.")

if __name__ == "__main__":
    main()