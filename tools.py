import os
import chromadb
from dotenv import load_dotenv
# --- IMPORT AGGIORNATI ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# -------------------------
from langchain_community.tools import DuckDuckGoSearchRun

# --- Costanti RAG (le stesse di ingest.py) ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "vc_pitch_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_rag_retriever(k_results=2):
    """
    Inizializza e restituisce un retriever RAG pronto all'uso.
    """
    print("Inizializzazione RAG retriever...")

    # 1. Inizializza il modello di embedding (con il nuovo import)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 2. Connettiti al ChromaDB ESISTENTE (con il nuovo import)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # 3. Definisci il "Retriever"
    retriever = vectordb.as_retriever(search_kwargs={"k": k_results})
    print("RAG retriever pronto.")
    return retriever


def get_web_search_tool():
    """
    Inizializza e restituisce il tool di ricerca web.
    """
    print("Inizializzazione Web search tool...")

    # --- CORREZIONE: Aggiunto 'region' e 'safesearch' ---
    web_search = DuckDuckGoSearchRun(
        max_results=3,
        region="it-it",  # Cerca dalla regione Italia / in lingua Italiana
        safesearch="moderate"  # Filtra contenuti espliciti
    )
    # ----------------------------------------------------

    print("Web search tool pronto.")
    return web_search


if __name__ == "__main__":
    # Piccolo test per assicurarsi che tutto funzioni
    load_dotenv()
    print("--- Test Trivial dei Tools ---")

    # Test RAG
    rag = get_rag_retriever()
    test_query_rag = "team di AstraBio"
    print(f"\nTest RAG con query: '{test_query_rag}'")
    try:
        results = rag.invoke(test_query_rag)
        print(f"Risultati RAG: {len(results)} chunk trovati. (Corretto)")
    except Exception as e:
        print(f"Errore test RAG: {e}")

    # Test Web
    web = get_web_search_tool()
    test_query_web = "Cosa è LangGraph?"
    print(f"\nTest Web con query: '{test_query_web}'")
    try:
        results_web = web.invoke(test_query_web)
        print(f"Risultati Web (primi 150 caratteri): {results_web[:150]}...")
    except Exception as e:
        print(f"Errore test Web: {e}")