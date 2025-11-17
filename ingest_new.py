import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Definiamo le cartelle
DOCS_DIR = "input_docs"  # La cartella che hai già creato
STORAGE_DIR = "./pitch_db" # Creeremo un DB vettoriale separato
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Il nostro modello locale affidabile

def main():
    """
    Ingestione manuale (robusta) dei pitch deck.
    """
    print(f"--- Avvio Ingestione RAG da '{DOCS_DIR}' ---")

    if not os.path.exists(DOCS_DIR):
        print(f"Errore: La cartella '{DOCS_DIR}' non è stata trovata.")
        print("Azione richiesta: Crea la cartella e aggiungi 'pitch_Perplexity.txt'.")
        return

    # 1. Carica i documenti
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    print("Caricamento documenti...")
    documents = loader.load()

    if not documents:
        print("Nessun documento .txt trovato.")
        return

    # 2. Splitta i documenti
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    print("Splitting dei documenti...")
    chunks = text_splitter.split_documents(documents)
    print(f"Creati {len(chunks)} chunk.")

    # 3. Inizializza Embedding
    print(f"Caricamento modello embedding: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 4. Inizializza ChromaDB e salva
    print(f"Creazione database vettoriale in '{STORAGE_DIR}'...")
    if os.path.exists(STORAGE_DIR):
        print("Rimuovo il vecchio DB...")
        import shutil
        shutil.rmtree(STORAGE_DIR)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=STORAGE_DIR
    )

    print(f"\n--- ✅ Ingestione Completata ---")
    print(f"I documenti sono stati indicizzati e salvati in '{STORAGE_DIR}'.")
    print("Ora puoi eseguire 'analyzer.py'.")

if __name__ == "__main__":
    main()