import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# IMPORT MODIFICATO: Usiamo HuggingFace invece di Google per gli embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Costanti ---
INPUT_DIR = "input_docs"
CHROMA_DB_PATH = "./chroma_db"
# MODELLO MODIFICATO: Usiamo un modello locale open source
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    """
    Funzione principale per l'ingestione dei documenti.
    Carica, splitta e salva i documenti in ChromaDB.
    """
    print("Avvio del processo di ingestione...")

    # 1. Carica i documenti .txt e .pdf separatamente
    print(f"Caricamento documenti da '{INPUT_DIR}'...")

    # Loader per i file .txt
    loader_txt = DirectoryLoader(
        INPUT_DIR,
        glob="**/*.txt",  # Cerca solo i .txt
        loader_cls=TextLoader,  # Usa il loader per il testo
        show_progress=True
    )
    # Loader per i file .pdf
    loader_pdf = DirectoryLoader(
        INPUT_DIR,
        glob="**/*.pdf",  # Cerca solo i .pdf
        loader_cls=PyPDFLoader,  # Usa il loader per i PDF
        show_progress=True
    )

    print("Caricamento file .txt...")
    documents_txt = loader_txt.load()
    print("Caricamento file .pdf...")
    documents_pdf = loader_pdf.load()

    # Uniamo le due liste di documenti
    documents = documents_txt + documents_pdf

    if not documents:
        print("Nessun documento trovato. Controlla la cartella 'input_docs'.")
        return

    print(f"Caricati {len(documents)} documenti.")

    # 2. Splitta i documenti in "chunk" più piccoli
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    print("Splitting dei documenti in chunk...")
    chunks = text_splitter.split_documents(documents)
    print(f"Creati {len(chunks)} chunk di testo.")

    # 3. Inizializza il modello di embedding (LOCALE)
    print(f"Inizializzazione modello embedding locale: {EMBEDDING_MODEL}")
    # Questa è la nuova riga: non serve API key
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}  # Forza l'uso della CPU
    )

    # 4. Inizializza ChromaDB e salva i chunk
    print(f"Inizializzazione ChromaDB (percorso: '{CHROMA_DB_PATH}')...")

    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection_name = "vc_pitch_docs"

    try:
        if collection_name in [c.name for c in client.list_collections()]:
            print(f"Collezione '{collection_name}' esistente. La elimino per re-ingestione.")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Errore durante la pulizia della collezione: {e}")

    print(f"Creazione collezione '{collection_name}' e salvataggio dei chunk...")

    # Questa chiamata ora userà il modello locale (HuggingFace)
    vectordb = Chroma.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    print(f"Ingestione completata! {len(chunks)} chunk salvati in '{collection_name}'.")


if __name__ == "__main__":
    main()