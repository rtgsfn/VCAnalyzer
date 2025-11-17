import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Importa i nostri tool
from tools import get_rag_retriever, get_web_search_tool
from graph import GraphTool


# --- 1. Definizione dello STATO (Il "Cervello" del Grafo) ---

class AgentState(TypedDict):
    """
    Lo stato definisce la "memoria" del nostro agente.
    Viene passato tra ogni nodo del grafo.
    """
    user_request: str  # La richiesta originale dell'utente
    startup_name: str  # Il nome della startup identificata
    founders_names: List[str]  # I nomi dei fondatori trovati
    doc_report: str  # Il report dall'analisi RAG (documenti)
    graph_report: str  # Il report dall'analisi Graph (Neo4j)
    web_report: str  # Il report dall'analisi Web
    final_summary: str  # Il report finale sintetizzato


# --- 2. Inizializzazione dei Tool e dell'LLM ---

# Carica le variabili d'ambiente (.env)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inizializza l'LLM (Gemini 2.5 Pro)
# Questo LLM verrà usato per tutte le task "intelligenti"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # O 1.5-pro se 2.5 non è disponibile
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1  # Molto fattuale
)

# Inizializza gli "Specialisti"
print("Inizializzazione tool... (attendere)")
rag_retriever = get_rag_retriever(k_results=3)
web_search_tool = get_web_search_tool()

# Inizializza il tool Neo4j
graph_tool = GraphTool(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)
print("Tool inizializzati.")


# --- 3. Definizione dei NODI del Grafo (Le Azioni) ---

def planner_node(state: AgentState) -> AgentState:
    """
    Nodo 1: Il "Planner".
    Analizza la richiesta, estrae le entità chiave (nome startup)
    e pianifica i passi successivi.
    """
    print("\n--- NODO: Planner ---")
    user_request = state["user_request"]

    # Prompt per estrarre il nome della startup
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un analista. Il tuo compito è estrarre il nome della startup dalla richiesta dell'utente. Rispondi solo con il nome."),
        ("user", "{request}")
    ])

    # Usiamo un parser per forzare l'LLM a rispondere solo con una stringa
    extractor_chain = prompt_template | llm | StrOutputParser()

    startup_name = extractor_chain.invoke({"request": user_request})

    print(f"Startup identificata: {startup_name}")

    return {"startup_name": startup_name}


def rag_node(state: AgentState) -> AgentState:
    """
    Nodo 2: L'Agente RAG (Analisi Documenti).
    Interroga i documenti caricati (pitch deck) sulla startup.
    """
    print("\n--- NODO: RAG (Documenti) ---")
    startup_name = state["startup_name"]

    # 1. Recupera i chunk rilevanti
    query_rag = f"Cosa dicono i documenti caricati (pitch deck) sul team e sul background dei fondatori della startup '{startup_name}'?"
    documents = rag_retriever.invoke(query_rag)

    # Formatta i documenti per l'LLM
    doc_context = "\n---\n".join([doc.page_content for doc in documents])

    # 2. Invia i chunk all'LLM per una sintesi
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un analista. Sintetizza il seguente contesto preso da un pitch deck, focalizzandoti su team e fondatori. Se non trovi info, rispondi 'Nessuna informazione trovata nel pitch deck.'"),
        ("user", "Contesto:\n{context}")
    ])

    summarizer_chain = prompt_template | llm | StrOutputParser()
    doc_report = summarizer_chain.invoke({"context": doc_context})

    print(f"Report RAG: {doc_report[:100]}...")
    return {"doc_report": doc_report}


def graph_node(state: AgentState) -> AgentState:
    """
    Nodo 3: L'Agente Graph (Relazioni).
    Trova i fondatori nel KG e poi cerca la loro storia passata.
    """
    print("\n--- NODO: Graph (Neo4j) ---")
    startup_name = state["startup_name"]

    # Step A: Trova i nomi dei fondatori
    founder_names = graph_tool.find_founders_by_startup(startup_name)
    if not founder_names:
        report = "Nessun fondatore trovato nel Knowledge Graph."
        print(report)
        return {"graph_report": report, "founders_names": []}

    print(f"Fondatori trovati: {founder_names}")

    # Step B: Per ogni fondatore, cerca la sua storia
    reports = []
    for name in founder_names:
        details = graph_tool.find_person_details(name)
        reports.append(details)

    full_report = "\n".join(reports)

    print(f"Report Graph: {full_report[:100]}...")
    return {"graph_report": full_report, "founders_names": founder_names}


def web_node(state: AgentState) -> AgentState:
    """
    Nodo 4: L'Agente RAG (Web Search).
    Cerca notizie e informazioni pubbliche sui fondatori trovati.
    """
    print("\n--- NODO: Web Search ---")
    founders_names = state["founders_names"]

    if not founders_names:
        report = "Ricerca web saltata: nessun fondatore identificato."
        print(report)
        return {"web_report": report}

    reports = []
    for name in founders_names:
        print(f"Ricerca web per: {name}")
        query_web = f"Notizie recenti e background di {name}, con focus su fallimenti passati o 'red flags' gestionali."
        # Usiamo il tool DuckDuckGo
        search_results = web_search_tool.invoke(query_web)

        # Sintetizziamo i risultati della ricerca
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "Sei un analista investigativo. Riassumi i seguenti risultati di ricerca web, cercando specificamente 'red flags' o problemi gestionali passati. Ignora risultati irrilevanti."),
            ("user", "Risultati per {name}:\n{results}")
        ])
        summarizer_chain = prompt_template | llm | StrOutputParser()

        summary = summarizer_chain.invoke({"name": name, "results": search_results})
        reports.append(f"Report Web per {name}:\n{summary}")

    full_report = "\n---\n".join(reports)
    print(f"Report Web: {full_report[:100]}...")
    return {"web_report": full_report}


def synthesis_node(state: AgentState) -> AgentState:
    """
    Nodo 5: Il "Sintetizzatore".
    Prende tutti i report e genera il riassunto finale per il VC.
    """
    print("\n--- NODO: Synthesis ---")

    # Raccoglie tutti i pezzi di informazione
    context = f"""
    Richiesta Utente: {state['user_request']}

    1. Analisi Pitch Deck (RAG):
    {state['doc_report']}

    2. Analisi Relazionale (Knowledge Graph):
    {state['graph_report']}

    3. Analisi Fonti Pubbliche (Web):
    {state['web_report']}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un assistente VC senior. Il tuo compito è generare un'analisi del rischio concisa "
         "per un team di una startup. Integra le informazioni dal pitch deck, dal knowledge graph e dal web. "
         "Identifica i punti di forza dichiarati, i fatti relazionali (storia passata) e i rischi semantici (notizie negative). "
         "Sii fattuale e vai dritto al punto. Concludi con un 'RISCHIO RILEVATO' se ne trovi uno."),
        ("user", "Genera il report finale basandoti su queste informazioni:\n{context}")
    ])

    synthesis_chain = prompt_template | llm | StrOutputParser()
    final_summary = synthesis_chain.invoke({"context": context})

    print("Sintesi finale generata.")
    return {"final_summary": final_summary}


# --- 4. Costruzione del Grafo (LangGraph) ---

print("Costruzione del grafo (LangGraph)...")
workflow = StateGraph(AgentState)

# Aggiungi i nodi
workflow.add_node("planner", planner_node)
workflow.add_node("rag_docs", rag_node)
workflow.add_node("graph_db", graph_node)
workflow.add_node("web_search", web_node)
workflow.add_node("synthesis", synthesis_node)

# Definisci gli archi (il flusso)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "rag_docs")
workflow.add_edge("rag_docs", "graph_db")
workflow.add_edge("graph_db", "web_search")
workflow.add_edge("web_search", "synthesis")
workflow.add_edge("synthesis", END)

# Compila il grafo
app = workflow.compile()
print("Grafo compilato. Sistema pronto.")


# --- 5. Esecuzione del Grafo ---

def run_analysis(request: str):
    """
    Funzione helper per eseguire il grafo con una richiesta.
    """
    # Questo è l'input iniziale
    inputs = {"user_request": request}

    # 'stream' esegue il grafo passo dopo passo e ci mostra i risultati intermedi
    print(f"\n\n--- AVVIO ANALISI PER: '{request}' ---")
    config = RunnableConfig(recursion_limit=50)  # Aumenta il limite di ricorsione

    try:
        # Usiamo 'stream' per vedere il progresso
        for chunk in app.stream(inputs, config=config):
            # 'chunk' conterrà l'output dell'ultimo nodo eseguito
            # Lo stampiamo per vedere cosa sta succedendo
            print(f"Stato aggiornato: {list(chunk.keys())[0]}")
            # print(chunk) # Debug: stampa l'intero stato

        # Alla fine, recuperiamo lo stato finale
        final_state = app.invoke(inputs, config=config)

        print("\n\n--- ✅ ANALISI COMPLETATA ---")
        print("Report Finale per il VC:")
        print("---------------------------------")
        print(final_state["final_summary"])

    except Exception as e:
        print(f"\n--- ❌ ERRORE DURANTE L'ESECUZIONE ---")
        print(e)
    finally:
        # Chiudi la connessione a Neo4j
        graph_tool.close()


if __name__ == "__main__":
    # La richiesta d'esempio che abbiamo definito all'inizio
    richiesta_esempio = "Voglio un'analisi del rischio sul team della startup 'AstraBio'."

    run_analysis(richiesta_esempio)