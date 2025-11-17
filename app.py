import streamlit as st
import time
import os
import tempfile
from typing import Tuple, List
import pandas as pd

# Streamlit e GUI
from streamlit_agraph import agraph, Node, Edge, Config

# LangChain - Documenti e RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# I nostri Script
try:
    from analyzer import AgenticKRAG
    from extractor import KnowledgeGraph, DocumentClaims, Claim
    from graph import GraphTool
    from scraper import scrape_article_text
except ImportError as e:
    st.error(f"❌ Errore Critico: Impossibile importare i moduli. Dettagli: {e}")
    st.stop()

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

MODEL_OPTIONS = {
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b"
    ],
    "Perplexity": [
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
        "mixtral-8x7b-instruct"
    ],
    "Google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ],
    "Ollama (Locale)": [
        "llama3:8b-instruct",
        "gemma:7b-instruct",
        "mixtral:8x7b-instruct"
    ]
}


# ============================================================================
# FUNZIONI DI CACHE
# ============================================================================

@st.cache_resource(show_spinner="🔧 Inizializzazione agente...")
def get_agent_instance(provider: str, model_name: str) -> AgenticKRAG:
    """Inizializza e mette in cache l'istanza dell'Agente KRAG (senza callback GUI)."""
    try:
        # Nessun callback qui: verrà assegnato dopo, fuori da Streamlit cache
        agent = AgenticKRAG(provider=provider, model_name=model_name)
        return agent
    except Exception as e:
        st.error(f"❌ Errore critico durante l'inizializzazione: {e}")
        return None


@st.cache_resource(show_spinner="📦 Caricamento modello embedding...")
def load_embedding_resources():
    """Carica e mette in cache le risorse per il RAG."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return embeddings, splitter


@st.cache_resource(show_spinner="📄 Elaborazione documenti...")
def process_documents(files_or_text, _embeddings, _splitter) -> Tuple[object, str]:
    """Processa file/testo e restituisce (retriever, full_text)."""
    all_documents = []
    full_text_list = []

    if isinstance(files_or_text, str):  # Testo incollato
        full_text = files_or_text
        all_documents = [Document(page_content=full_text, metadata={"source": "pasted_text"})]
        full_text_list.append(full_text)
    elif isinstance(files_or_text, list):  # File caricati
        for uploaded_file in files_or_text:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            try:
                if tmp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif tmp_file_path.endswith('.txt'):
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                else:
                    st.warning(f"⚠️ File '{uploaded_file.name}' ignorato (formato non supportato)")
                    continue
                documents = loader.load()
                all_documents.extend(documents)
                full_text_list.append("\n\n".join([doc.page_content for doc in documents]))
            except Exception as e:
                st.error(f"❌ Errore elaborazione '{uploaded_file.name}': {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
        full_text = "\n\n--- NUOVO DOCUMENTO ---\n\n".join(full_text_list)

    if not all_documents:
        st.error("❌ Nessun documento valido processato")
        return None, None

    chunks = _splitter.split_documents(all_documents)
    vectordb = Chroma.from_documents(documents=chunks, embedding=_embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3}), full_text


# ============================================================================
# FUNZIONI VISUALIZZAZIONE
# ============================================================================
def make_status_callback(container):
    """Wrapper per adattare (message, level) alle chiamate Streamlit."""
    def _callback(message: str, level: str = "info"):
        if level == "error":
            container.error(message)
        elif level == "warning":
            container.warning(message)
        elif level == "success":
            container.success(message)
        else:
            container.info(message)
    return _callback


def render_fact_checking_table(claims_data: List[dict]):
    """Renderizza tabella fact-checking con styling."""
    if not claims_data:
        st.info("ℹ️ Nessuna affermazione fattuale estratta")
        return

    # Converti in DataFrame
    df = pd.DataFrame(claims_data)

    # Aggiungi icone
    status_icons = {
        "VERIFICATA": "✅",
        "FALSA": "❌",
        "PARZIALMENTE VERIFICATA": "⚠️",
        "NON VERIFICABILE": "❓"
    }
    df["Status"] = df["status"].apply(lambda x: f"{status_icons.get(x, '?')} {x}")

    # Riordina colonne
    df_display = df[["soggetto", "affermazione", "Status", "prove"]].copy()
    df_display.columns = ["Soggetto", "Affermazione", "Verdetto", "Prove (Estratto)"]

    # Accorcia prove per leggibilità
    df_display["Prove (Estratto)"] = df_display["Prove (Estratto)"].apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x)

    # Statistiche
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        verified = len([c for c in claims_data if c["status"] == "VERIFICATA"])
        st.metric("✅ Verificate", verified)
    with col2:
        false = len([c for c in claims_data if c["status"] == "FALSA"])
        st.metric("❌ False", false)
    with col3:
        partial = len([c for c in claims_data if c["status"] == "PARZIALMENTE VERIFICATA"])
        st.metric("⚠️ Parziali", partial)
    with col4:
        unverifiable = len([c for c in claims_data if c["status"] == "NON VERIFICABILE"])
        st.metric("❓ Non Verificabili", unverifiable)

    st.dataframe(df_display, width='stretch', hide_index=True)


def render_knowledge_graph(nodes_data, edges_data, entity_name):
    """Renderizza grafo con styling migliorato."""
    if not nodes_data:
        st.info(f"ℹ️ Nessun dato nel Knowledge Graph per '{entity_name}'")
        return

    # Palette colori migliorata
    color_map = {
        "Startup": "#FF6B6B",  # Rosso corallo
        "Persona": "#4ECDC4",  # Turchese
        "Investitore": "#45B7D1",  # Blu chiaro
        "Entita": "#95E1D3",  # Verde acqua
    }

    nodes = []
    for n in nodes_data:
        # Determina il tipo dalla label
        node_type = n["label"].split("\n")[0] if "\n" in n["label"] else "Entita"
        color = color_map.get(node_type, "#95E1D3")

        # Colore speciale per nodo target
        if n["id"] == entity_name:
            color = "#FFD93D"  # Giallo oro per il focus
            size = 25
        else:
            size = 18

        nodes.append(Node(
            id=n["id"],
            label=n["label"],
            color=color,
            size=size,
            font={"size": 14, "color": "#2C3E50"}
        ))

    edges = []
    for e in edges_data:
        edges.append(Edge(
            source=e["source"],
            target=e["target"],
            label=e["label"],
            color="#7F8C8D",
            width=2
        ))

    config = Config(
        width=900,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7DC6F",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True}
    )

    st.markdown(f"### 🕸️ Knowledge Graph per **{entity_name}**")
    st.caption("🟡 Giallo = Entità Focus | 🔴 Rosso = Startup | 🔵 Blu = Investitore | 🟢 Verde = Persona")

    agraph(nodes=nodes, edges=edges, config=config)


def render_metadata_sidebar(metadata: dict):
    """Renderizza sidebar con metadata analisi."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌟 Confidence Dashboard")

        if "error" in metadata:
            st.error(f"❌ {metadata['error']}")
            return

        # 1. Confidence Deduzione Entità
        conf_entity = metadata.get('confidence_entity', 'low')
        if conf_entity == 'high':
            st.metric("🎯 Confidenza Entità", "Alta", "✅")
        elif conf_entity == 'medium':
            st.metric("🎯 Confidenza Entità", "Media", "⚠️")
        else:
            st.metric("🎯 Confidenza Entità", "Bassa", "❌")

        # 2. Confidence Fact-Checking
        fact_check_score = metadata.get('confidence_fact_check_score', 0)
        st.progress(fact_check_score, text=f"Score Verificabilità: {fact_check_score:.0%}")
        st.caption("Verificate (100%) + Parziali (50%) / Totali")

        # 3. Confidence Grafo
        graph_sources = metadata.get('confidence_graph_sources', 0)
        st.metric("🕸️ Corroborazione Grafo", f"{graph_sources} Nodi Trovati")
        st.caption("Nodi trovati nel grafo pubblico (esclusa l'entità focus)")

        st.markdown("---")
        st.markdown("### 📊 Statistiche Analisi")

        st.metric("⏱️ Tempo Analisi", f"{metadata.get('analysis_time_seconds', 0):.1f}s")
        st.metric("📝 Claims Totali", metadata.get('claims_total', 0))

        # Breakdown claims (questo va bene come prima)
        with st.expander("📊 Breakdown Fact-Checking"):
            st.write(f"✅ Verificate: {metadata.get('claims_verified', 0)}")
            st.write(f"❌ False: {metadata.get('claims_false', 0)}")
            st.write(f"⚠️ Parziali: {metadata.get('claims_partial', 0)}")
            st.write(f"❓ Non Verificabili: {metadata.get('claims_unverifiable', 0)}")

        st.metric("🕸️ Nodi Grafo Totali", metadata.get('graph_nodes', 0))
        st.metric("🔗 Relazioni Grafo Totali", metadata.get('graph_edges', 0))


# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.set_page_config(
    page_title="VC KRAG Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🎯 Agentic VC KRAG Analyzer")
st.markdown("**Knowledge-Retrieval Augmented Generation** per Due Diligence Investigativa")
st.markdown("---")

# Sidebar - Configurazione
with st.sidebar:
    st.header("⚙️ Configurazione")

    provider = st.selectbox("🤖 Provider LLM:", list(MODEL_OPTIONS.keys()))
    available_models = MODEL_OPTIONS.get(provider, [])

    if not available_models:
        st.error("❌ Provider non valido")
        st.stop()

    model_name = st.selectbox("📦 Modello:", available_models)

    st.markdown("---")
    st.markdown("### 📖 Guida Rapida")
    st.info("""
    **Step 1**: Carica documenti (pitch deck, memo, etc.)

    **Step 2**: Specifica l'entità da analizzare

    **Step 3**: Avvia l'analisi investigativa

    Il sistema:
    - ✅ Verifica claim del documento
    - 🕸️ Cerca fatti pubblici
    - 📊 Genera report rischi/fattibilità
    """)

# Caricamento Risorse
try:
    embeddings, splitter = load_embedding_resources()
except Exception as e:
    st.error(f"❌ Errore caricamento embeddings: {e}")
    st.stop()

# Input Principale
st.markdown("## 📂 Step 1: Carica Documenti")

tab1, tab2 = st.tabs(["📁 Upload Files", "📋 Paste Text"])

with tab1:
    uploaded_files = st.file_uploader(
        "Carica fino a 5 documenti (.txt o .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

with tab2:
    pasted_text = st.text_area(
        "Incolla il testo del documento:",
        height=250,
        placeholder="Es. pitch deck, investment memo, company overview..."
    )

st.markdown("---")
st.markdown("## 🎯 Step 2: Entità da Analizzare")

# Container per deduzione automatica
auto_detect_container = st.container()

entita_utente = st.text_input(
    "Quale entità vuoi analizzare?",
    placeholder="Es. 'Figure AI', 'OpenAI', 'Sam Altman'",
    help="Lascia vuoto per deduzione automatica dal documento"
)

# Checkbox per conferma manuale
use_auto_detect = st.checkbox(
    "🤖 Rileva automaticamente l'entità dal documento",
    value=False,
    help="Il sistema analizzerà il documento per identificare l'entità principale"
)

is_deep_search = st.toggle(
    "🚀 Attiva Deep Search (Approfondita)",
    value=False,
    help="Usa la modalità di ricerca avanzata. È più lenta (fino a 2 minuti) e più costosa, ma analizza un numero maggiore di fonti (fino a 20) per un grafo e metriche più completi."
)

st.markdown("---")

# ============================================================================
# ESECUZIONE ANALISI
# ============================================================================
# [Mantieni tutti gli import e le funzioni precedenti fino al bottone "Avvia Analisi"]
# Sostituisci solo la sezione di esecuzione analisi con questo codice:

if st.button("🚀 Avvia Analisi Investigativa", type="primary", width='stretch'):

    # Validazione input
    input_data = None
    if uploaded_files:
        input_data = uploaded_files
    elif pasted_text:
        input_data = pasted_text
    else:
        st.error("❌ Carica almeno un documento o incolla del testo")
        st.stop()

    # Container per status e risultati progressivi
    status_container = st.empty()
    progress_bar = st.progress(0)

    # Placeholder per risultati progressivi
    metrics_placeholder = st.empty()
    factcheck_placeholder = st.empty()
    metrics_analysis_placeholder = st.empty()
    risk_placeholder = st.empty()
    feasibility_placeholder = st.empty()
    summary_placeholder = st.empty()
    graph_placeholder = st.empty()

    try:
        # ====================================================================
        # SETUP INIZIALE
        # ====================================================================
        status_container.info("🔧 Inizializzazione agente...")
        agent = get_agent_instance(provider, model_name)
        if not agent:
            st.stop()

        # ora che l'agente esiste, gli assegniamo il callback compatibile
        agent.status_callback = make_status_callback(status_container)

        progress_bar.progress(5)
        status_container.info("📄 Processamento documenti...")
        doc_retriever, document_text = process_documents(input_data, embeddings, splitter)

        if not doc_retriever or not document_text:
            st.stop()

        progress_bar.progress(10)

        # Deduzione automatica entità (se richiesto)
        entity_to_analyze = entita_utente.strip()

        if use_auto_detect or not entity_to_analyze:
            status_container.info("🔍 Deduzione automatica entità dal documento...")
            entity_analysis = agent.deduce_entity_from_document(document_text)

            if entity_analysis["entity_found"]:
                suggested_entity = entity_analysis["entity_name"]
                with auto_detect_container:
                    st.success(f"🎯 **Entità Rilevata**: {suggested_entity}")
                    st.caption(
                        f"📊 Confidenza: {entity_analysis['confidence'].upper()} | 📝 {entity_analysis['context']}")

                if not entity_to_analyze:
                    entity_to_analyze = suggested_entity
                    st.info(f"✅ Analisi per: **{entity_to_analyze}**")

        if not entity_to_analyze:
            st.error("❌ Specifica un'entità da analizzare")
            st.stop()

        progress_bar.progress(15)

        # ====================================================================
        # ESECUZIONE ANALISI CON STREAMING PROGRESSIVO
        # ====================================================================

        status_container.info(f"🚀 Avvio analisi per '{entity_to_analyze}'...")
        start_time = time.time()

        # Usa la nuova funzione streaming
        result = agent.run_full_analysis_streaming(entity_to_analyze, document_text, doc_retriever, is_deep_search=is_deep_search)

        if "error" in result:
            st.error(f"❌ Errore: {result['error']}")
            st.stop()

        elapsed = time.time() - start_time
        progress_bar.progress(20)

        # ====================================================================
        # RENDERING PROGRESSIVO RISULTATI
        # ====================================================================

        st.markdown("---")
        st.markdown("# 📊 Investment Analysis Report")
        st.caption(f"Analisi completata in {elapsed:.1f} secondi")

        # ====================================================================
        # 1. METRICHE VC (Prime a essere visualizzate)
        # ====================================================================
        progress_bar.progress(30)

        with metrics_placeholder.container():
            st.markdown("## 📊 Metriche VC Estratte")

            vc_metrics = result.get("vc_metrics")
            if vc_metrics:
                # Crea tab per categorie metriche
                tab_saas, tab_traction, tab_market, tab_team, tab_fundraising = st.tabs([
                    "💰 SaaS", "📈 Traction", "🌍 Market", "👥 Team", "💵 Fundraising"
                ])

                with tab_saas:
                    if vc_metrics.saas_metrics:
                        m = vc_metrics.saas_metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if m.arr: st.metric("ARR", f"${m.arr}M")
                            if m.revenue_growth_rate: st.metric("Growth YoY", f"{m.revenue_growth_rate}%")
                        with col2:
                            if m.ltv_cac_ratio: st.metric("LTV/CAC", f"{m.ltv_cac_ratio}x")
                            if m.net_retention_rate: st.metric("Net Retention", f"{m.net_retention_rate}%")
                        with col3:
                            if m.runway_months: st.metric("Runway", f"{m.runway_months} mesi")
                            if m.rule_of_40: st.metric("Rule of 40", f"{m.rule_of_40}")
                    else:
                        st.info("ℹ️ Nessuna metrica SaaS estratta")

                with tab_traction:
                    if vc_metrics.traction_metrics:
                        m = vc_metrics.traction_metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            if m.total_users: st.metric("Utenti Totali", f"{m.total_users:,}")
                            if m.paying_customers: st.metric("Clienti Paganti", f"{m.paying_customers:,}")
                        with col2:
                            if m.user_growth_rate: st.metric("Crescita MoM", f"{m.user_growth_rate}%")
                            if m.nps_score: st.metric("NPS Score", m.nps_score)
                    else:
                        st.info("ℹ️ Nessuna metrica traction estratta")

                with tab_market:
                    if vc_metrics.market_metrics:
                        m = vc_metrics.market_metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if m.tam: st.metric("TAM", f"${m.tam}B")
                        with col2:
                            if m.sam: st.metric("SAM", f"${m.sam}B")
                        with col3:
                            if m.som: st.metric("SOM", f"${m.som}M")
                    else:
                        st.info("ℹ️ Nessuna metrica market estratta")

                with tab_team:
                    if vc_metrics.team_metrics and vc_metrics.team_metrics.founders:
                        for founder in vc_metrics.team_metrics.founders:
                            st.markdown(f"**{founder.name}** - {founder.role}")
                            if founder.background:
                                st.caption(f"📝 {founder.background}")
                            st.markdown("---")
                    else:
                        st.info("ℹ️ Nessuna info team estratta")

                with tab_fundraising:
                    if vc_metrics.fundraising_metrics and vc_metrics.fundraising_metrics.rounds:
                        for round_data in vc_metrics.fundraising_metrics.rounds:
                            st.markdown(f"**{round_data.stage.value}** - ${round_data.amount}M")
                            if round_data.lead_investor:
                                st.caption(f"Lead: {round_data.lead_investor}")
                            st.markdown("---")
                    else:
                        st.info("ℹ️ Nessuna info fundraising estratta")

        # ====================================================================
        # 2. FACT-CHECKING TABLE
        # ====================================================================
        progress_bar.progress(40)

        with factcheck_placeholder.container():
            st.markdown("## ✅ Fact-Checking Affermazioni")
            render_fact_checking_table(result.get("fact_checking_table", []))

        # ====================================================================
        # 3. METRICS ANALYSIS
        # ====================================================================
        progress_bar.progress(50)

        with metrics_analysis_placeholder.container():
            #st.markdown("## 📊 Analisi Metriche vs Benchmark")
            st.markdown(result.get("metrics_analysis", "_Analisi in corso..._"))

        # ====================================================================
        # 4. RISK ANALYSIS
        # ====================================================================
        progress_bar.progress(65)

        with risk_placeholder.container():
            #st.markdown("## 🚩 Risk Analysis")
            st.markdown(result.get("risk_analysis", "_Analisi in corso..._"))

        # ====================================================================
        # 5. FEASIBILITY ANALYSIS
        # ====================================================================
        progress_bar.progress(80)

        with feasibility_placeholder.container():
            #st.markdown("## ✅ Feasibility Analysis")
            st.markdown(result.get("feasibility_analysis", "_Analisi in corso..._"))

        # ====================================================================
        # 6. EXECUTIVE SUMMARY
        # ====================================================================
        progress_bar.progress(90)

        with summary_placeholder.container():
            st.markdown("## 📋 Executive Summary & Recommendation")
            st.markdown(result.get("executive_summary", "_Generazione in corso..._"))

        # ====================================================================
        # 7. KNOWLEDGE GRAPH
        # ====================================================================
        progress_bar.progress(95)

        with graph_placeholder.container():
            st.markdown("## 🕸️ Knowledge Graph")
            render_knowledge_graph(
                result["graph_data"]["nodes"],
                result["graph_data"]["edges"],
                entity_to_analyze
            )

        # ====================================================================
        # METADATA SIDEBAR
        # ====================================================================
        progress_bar.progress(100)
        render_metadata_sidebar(result.get("metadata", {}))

        # ====================================================================
        # DOWNLOAD BUTTONS
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📥 Download Risultati")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Download Report Completo
            full_report = f"""# Investment Analysis Report: {entity_to_analyze}

{result.get('metrics_analysis', '')}

{result.get('risk_analysis', '')}

{result.get('feasibility_analysis', '')}

{result.get('executive_summary', '')}
"""
            st.download_button(
                label="📄 Report Completo (MD)",
                data=full_report,
                file_name=f"report_{entity_to_analyze.replace(' ', '_')}.md",
                mime="text/markdown"
            )

        with col2:
            # Download Fact-Checking
            if result.get("fact_checking_table"):
                df = pd.DataFrame(result["fact_checking_table"])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="✅ Fact-Check (CSV)",
                    data=csv,
                    file_name=f"factcheck_{entity_to_analyze.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

        with col3:
            # Download Graph Data
            import json

            graph_json = json.dumps(result.get("graph_data", {}), indent=2)
            st.download_button(
                label="🕸️ Graph Data (JSON)",
                data=graph_json,
                file_name=f"graph_{entity_to_analyze.replace(' ', '_')}.json",
                mime="application/json"
            )

        with col4:
            # Download Metriche
            if result.get("vc_metrics"):
                metrics_dict = result["vc_metrics"].model_dump()
                metrics_json = json.dumps(metrics_dict, indent=2, default=str)
                st.download_button(
                    label="📊 Metriche (JSON)",
                    data=metrics_json,
                    file_name=f"metrics_{entity_to_analyze.replace(' ', '_')}.json",
                    mime="application/json"
                )

        status_container.success(f"✅ Analisi completata in {elapsed:.1f} secondi!")

    except Exception as e:
        st.error(f"❌ Errore durante l'analisi: {e}")
        import traceback

        with st.expander("🔍 Dettagli Errore"):
            st.code(traceback.format_exc())

    finally:
        if 'agent' in locals() and agent:
            agent.close_graph_connection()

# Footer
st.markdown("---")
st.caption("🤖 Powered by Agentic KRAG | Built with LangChain, Neo4j, Streamlit")