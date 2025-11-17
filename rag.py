import os
import chromadb
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Costanti ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "vc_pitch_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-pro"  # O "gemini-1.5-pro-latest" se 2.5 non è ancora disponibile


def main():
    """
    Funzione principale per testare l'interrogazione RAG.
    """
    print("Avvio del test RAG...")

    # 1. Carica le API Key di Google
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY non trovata. Assicurati che sia nel file .env")

    # 2. Inizializza il modello di embedding (LO STESSO di ingest.py)
    print(f"Caricamento modello embedding: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 3. Connettiti al ChromaDB ESISTENTE
    print(f"Connessione a ChromaDB (percorso: '{CHROMA_DB_PATH}')...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    print("Connessione riuscita.")

    # 4. Inizializza l'LLM (Gemini)
    print(f"Caricamento LLM: {LLM_MODEL}")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.3  # Bassa "creatività" per risposte fattuali
    )

    # 5. Definisci il "Retriever"
    # Il retriever è l'oggetto che "recupera" i documenti
    # k=2 significa "trovami i 2 chunk più rilevanti"
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # 6. Definisci il Prompt Template per il RAG
    # Questo è il prompt che "istruisce" Gemini
    template = """
    Sei un assistente per un Venture Capitalist.
    Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto fornito.
    Se l'informazione non è nel contesto, rispondi 'Non ho trovato informazioni nel documento'.
    Non usare la tua conoscenza pregressa.

    CONTESTO:
    {context}

    DOMANDA:
    {question}

    RISPOSTA FATTUALE:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 7. Crea la "Chain" RAG
    # Questa è la sequenza di operazioni:
    # 1. Prendi la domanda (input)
    # 2. Passala al retriever per ottenere i "chunk" (context)
    # 3. Passa domanda e context al prompt
    # 4. Passa il prompt formattato all'LLM
    # 5. Ottieni l'output come stringa

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("--- Test RAG avviato. Digita 'exit' per uscire. ---")

    while True:
        # 8. Fai una domanda
        query = input("\nFai una domanda sui documenti: ")
        if query.lower() == 'exit':
            break

        if not query:
            continue

        print("Sto pensando...")
        # Invochiamo la chain con la nostra domanda
        answer = rag_chain.invoke(query)

        print("\nRISPOSTA:")
        print(answer)


if __name__ == "__main__":
    main()