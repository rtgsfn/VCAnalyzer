import streamlit as st
import time
import os
import tempfile
from typing import Tuple, List
import pandas as pd
import docx  # pip install python-docx
from pptx import Presentation  # pip install python-pptx

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
    from vc_metrics import VCMetricsProfile, PharmaMetricsProfile, REMetricsProfile, LegalMetricsProfile # Import per la tipizzazione corretta
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
        "mixtral:8x7b-instruct",
        "kimi-k2:1t-cloud"
    ]
}


# ============================================================================
# FUNZIONI DI UTILITÀ (ESTRAZIONE TESTO)
# ============================================================================

def extract_text_from_file(uploaded_file) -> str:
    """Estrae testo da PDF, DOCX, PPTX, TXT. Usato per i Requisiti."""
    text = ""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if suffix == '.pdf':
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
        elif suffix == '.txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            docs = loader.load()
            text = docs[0].page_content
        elif suffix in ['.docx', '.doc']:
            doc = docx.Document(tmp_file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif suffix in ['.pptx', '.ppt']:
            prs = Presentation(tmp_file_path)
            text_runs = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_runs.append(shape.text)
            text = "\n".join(text_runs)
    except Exception as e:
        st.error(f"❌ Errore lettura file {uploaded_file.name}: {e}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return text


# ============================================================================
# FUNZIONI DI CACHE
# ============================================================================

@st.cache_resource(show_spinner="🔧 Inizializzazione agente...")
def get_agent_instance(provider: str, model_name: str) -> AgenticKRAG:
    """Inizializza e mette in cache l'istanza dell'Agente KRAG."""
    try:
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


@st.cache_resource(show_spinner="📄 Elaborazione documenti (PDF, TXT, DOCX, PPTX)...")
def process_documents(files_or_text, _embeddings, _splitter) -> Tuple[object, str]:
    """Processa file eterogenei e restituisce (retriever, full_text)."""
    all_documents = []
    full_text_list = []

    if isinstance(files_or_text, str):  # Testo incollato
        full_text_list.append(files_or_text)
        all_documents = [Document(page_content=files_or_text, metadata={"source": "pasted_text"})]

    elif isinstance(files_or_text, list):  # Lista di file caricati
        for uploaded_file in files_or_text:
            # Usiamo la funzione helper per estrarre il testo grezzo
            extracted_text = extract_text_from_file(uploaded_file)

            if extracted_text:
                # --- MODIFICA IMPORTANTE: INSERIAMO IL NOME DEL FILE NEL TESTO ---
                # Questo permette all'LLM di citare la fonte nel report
                text_with_header = f"\n\n--- FILE: {uploaded_file.name} ---\n{extracted_text}"

                full_text_list.append(text_with_header)

                # Creiamo il Document object per LangChain
                all_documents.append(Document(
                    page_content=extracted_text,
                    metadata={"source": uploaded_file.name}
                ))

    if not all_documents:
        st.error("❌ Nessun documento valido processato")
        return None, None

    # Uniamo tutto il testo
    full_text = "\n".join(full_text_list)

    chunks = _splitter.split_documents(all_documents)
    vectordb = Chroma.from_documents(documents=chunks, embedding=_embeddings)

    return vectordb.as_retriever(search_kwargs={"k": 5}), full_text


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

    df = pd.DataFrame(claims_data)
    status_icons = {
        "VERIFICATA": "✅",
        "FALSA": "❌",
        "PARZIALMENTE VERIFICATA": "⚠️",
        "NON VERIFICABILE": "❓"
    }
    df["Status"] = df["status"].apply(lambda x: f"{status_icons.get(x, '?')} {x}")
    df_display = df[["soggetto", "affermazione", "Status", "prove"]].copy()
    df_display.columns = ["Soggetto", "Affermazione", "Verdetto", "Prove (Estratto)"]
    df_display["Prove (Estratto)"] = df_display["Prove (Estratto)"].apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x)

    # Statistiche
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("✅ Verificate", len([c for c in claims_data if c["status"] == "VERIFICATA"]))
    with col2: st.metric("❌ False", len([c for c in claims_data if c["status"] == "FALSA"]))
    with col3: st.metric("⚠️ Parziali", len([c for c in claims_data if c["status"] == "PARZIALMENTE VERIFICATA"]))
    with col4: st.metric("❓ Non Verif.", len([c for c in claims_data if c["status"] == "NON VERIFICABILE"]))

    st.dataframe(df_display, width='stretch', hide_index=True)


def render_knowledge_graph(nodes_data, edges_data, entity_name):
    """Renderizza grafo con styling migliorato."""
    if not nodes_data:
        st.info(f"ℹ️ Nessun dato nel Knowledge Graph per '{entity_name}'")
        return

    color_map = {"Startup": "#FF6B6B", "Persona": "#4ECDC4", "Investitore": "#45B7D1", "Entita": "#95E1D3"}
    nodes = []
    for n in nodes_data:
        node_type = n["label"].split("\n")[0] if "\n" in n["label"] else "Entita"
        color = color_map.get(node_type, "#95E1D3")
        size = 25 if n["id"] == entity_name else 18
        color = "#FFD93D" if n["id"] == entity_name else color

        nodes.append(Node(id=n["id"], label=n["label"], color=color, size=size, font={"size": 14, "color": "#2C3E50"}))

    edges = [Edge(source=e["source"], target=e["target"], label=e["label"], color="#7F8C8D", width=2) for e in
             edges_data]

    config = Config(width=900, height=600, directed=True, physics=True, hierarchical=False,
                    nodeHighlightBehavior=True, highlightColor="#F7DC6F", collapsible=False)

    st.markdown(f"### 🕸️ Knowledge Graph per **{entity_name}**")
    agraph(nodes=nodes, edges=edges, config=config)


def render_metadata_sidebar(metadata: dict):
    """Renderizza sidebar con metadata analisi."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌟 Confidence Dashboard")

        if "error" in metadata:
            st.error(f"❌ {metadata['error']}")
            return

        conf_entity = metadata.get('confidence_entity', 'low')
        if conf_entity == 'high':
            st.metric("🎯 Confidenza Entità", "Alta", "✅")
        elif conf_entity == 'medium':
            st.metric("🎯 Confidenza Entità", "Media", "⚠️")
        else:
            st.metric("🎯 Confidenza Entità", "Bassa", "❌")

        st.progress(metadata.get('confidence_fact_check_score', 0),
                    text=f"Score Verificabilità: {metadata.get('confidence_fact_check_score', 0):.0%}")
        st.metric("🕸️ Corroborazione Grafo", f"{metadata.get('confidence_graph_sources', 0)} Nodi Trovati")

        st.markdown("---")
        st.metric("⏱️ Tempo Analisi", f"{metadata.get('analysis_time_seconds', 0):.1f}s")
        st.metric("📝 Claims Totali", metadata.get('claims_total', 0))


# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.set_page_config(page_title="VC KRAG Analyzer", page_icon="🎯", layout="wide")

st.title("🎯 Agentic VC KRAG Analyzer")
st.markdown("**Knowledge-Retrieval Augmented Generation** per Due Diligence Investigativa")
st.markdown("---")

# Sidebar Config
with st.sidebar:
    st.header("⚙️ Configurazione")
    sector = st.selectbox(
        "🏢 Settore di Analisi",
        ["Venture Capital", "Real Estate", "Legal / M&A", "Pharma & Biotech"],
        index=0,
        help="Cambia la prospettiva dell'agente (es. Analista Finanziario vs Avvocato)"
    )

    st.divider()
    provider = st.selectbox("🤖 Provider LLM:", list(MODEL_OPTIONS.keys()))
    available_models = MODEL_OPTIONS.get(provider, [])
    model_name = st.selectbox("📦 Modello:", available_models)

try:
    embeddings, splitter = load_embedding_resources()
except Exception as e:
    st.error(f"❌ Errore caricamento risorse: {e}")
    st.stop()

# 1. Input Documenti Principali
st.markdown("## 📂 Step 1: Carica Documenti da Analizzare")
st.caption("Supporta PDF, DOCX, PPTX, TXT")
tab1, tab2 = st.tabs(["📁 Upload Files", "📋 Paste Text"])

with tab1:
    uploaded_files = st.file_uploader(
        "Carica documenti (Pitch deck, Memo, etc.)",
        type=["txt", "pdf", "docx", "doc", "pptx", "ppt"],
        accept_multiple_files=True
    )
with tab2:
    pasted_text = st.text_area("Incolla testo del documento", height=150)

st.markdown("---")

# 2. Configurazione Analisi
st.markdown("## 🎯 Step 2: Configurazione Analisi")

col_conf1, col_conf2 = st.columns([1, 2])

# Variabili di default (per evitare errori se non definite)
entita_utente = ""
requirements_text = None
is_deep_search = False
use_auto_detect = False

with col_conf1:
    analysis_mode = st.radio(
        "Tipo di Analisi:",
        ["Standard Due Diligence", "Matching Requisiti Tecnici"],
        help="Standard: Analisi rischi/opportunità. Requisiti: Verifica puntuale di una lista di necessità."
    )

# Logica Condizionale per la UI
if analysis_mode == "Matching Requisiti Tecnici":
    # --- MODALITÀ MATCHING (Requisiti visibili, Entità/Web nascosti) ---
    st.info("📋 **Modalità Requisiti**: Carica la lista delle necessità tecniche/funzionali.")
    req_tab1, req_tab2 = st.tabs(["📂 Carica File Requisiti", "✍️ Incolla Lista"])

    with req_tab1:
        req_file = st.file_uploader(
            "Carica file requisiti (.pdf, .docx, .pptx, .txt)",
            type=["txt", "pdf", "docx", "doc", "pptx", "ppt"],
            key="req_uploader"
        )
        if req_file:
            requirements_text = extract_text_from_file(req_file)
            if requirements_text:
                st.success(f"✅ Requisiti caricati ({len(requirements_text)} caratteri)")

    with req_tab2:
        req_input = st.text_area("Incolla qui la lista tecnica:", height=150,
                                 placeholder="- Supporto SSO SAML\n- Hosting On-Premise...")
        if req_input:
            requirements_text = req_input

    # Nella colonna 2 non mostriamo nulla in questa modalità
    with col_conf2:
        st.write("")

else:
    # --- MODALITÀ STANDARD (Entità/Web visibili, Requisiti nascosti) ---
    with col_conf2:
        entita_utente = st.text_input("Entità Target (Opzionale se rilevabile):", placeholder="Es. 'Figure AI'")
        use_auto_detect = st.checkbox("🤖 Rileva automaticamente l'entità", value=True)
        is_deep_search = st.toggle("🚀 Deep Search (Web)", value=False)

st.markdown("---")

# ============================================================================
# ESECUZIONE ANALISI (UNICO PULSANTE)
# ============================================================================

if st.button("🚀 Avvia Analisi", type="primary", width='stretch'):

    # 1. Validazione Input
    input_data = uploaded_files if uploaded_files else pasted_text
    if not input_data:
        st.error("❌ Manca il documento da analizzare (Step 1)")
        st.stop()

    if analysis_mode == "Matching Requisiti Tecnici" and not requirements_text:
        st.error("❌ Manca la lista dei requisiti per il matching.")
        st.stop()

    if analysis_mode == "Standard Due Diligence" and not entita_utente and not use_auto_detect:
        st.error("❌ Specifica un'entità da analizzare o abilita il rilevamento automatico.")
        st.stop()

    # Container per status e risultati progressivi
    status_container = st.empty()
    progress_bar = st.progress(0)

    # Placeholders
    metrics_placeholder = st.empty()
    results_placeholder = st.empty()
    graph_placeholder = st.empty()

    try:
        # 2. Inizializzazione
        status_container.info("🔧 Inizializzazione agente...")
        agent = get_agent_instance(provider, model_name)
        if not agent: st.stop()
        agent.status_callback = make_status_callback(status_container)

        # 3. Processamento Documenti
        progress_bar.progress(10)
        status_container.info("📄 Processamento documenti...")
        doc_retriever, document_text = process_documents(input_data, embeddings, splitter)

        if not doc_retriever or not document_text: st.stop()

        # 4. Deduzione Entità
        entity_to_analyze = entita_utente.strip()
        should_deduce = False

        if analysis_mode == "Matching Requisiti Tecnici":
            # FIX: In matching mode NON deduciamo nulla e usiamo un placeholder generico
            should_deduce = False
            entity_to_analyze = "Gap Analysis"
        elif use_auto_detect and not entity_to_analyze:
            # In standard mode, deduciamo se richiesto
            should_deduce = True

        if should_deduce:
            status_container.info("🔍 Deduzione entità in corso...")
            deduction = agent.deduce_entity_from_document(document_text)
            if deduction["entity_found"]:
                entity_to_analyze = deduction["entity_name"]
                status_container.success(f"✅ Identificato: {entity_to_analyze}")
            else:
                st.error("Impossibile identificare l'entità automaticamente. Inseriscila manualmente.")
                st.stop()

        # 5. Esecuzione Analisi
        start_time = time.time()
        progress_bar.progress(20)

        # Forza deep_search a False in modalità matching
        actual_deep_search = False if analysis_mode == "Matching Requisiti Tecnici" else is_deep_search

        result = agent.run_full_analysis_streaming(
            entity_to_analyze,
            document_text,
            doc_retriever,
            is_deep_search=actual_deep_search,
            requirements_text=requirements_text,
            sector=sector
        )

        if "error" in result:
            st.error(f"❌ Errore: {result['error']}")
            st.stop()

        progress_bar.progress(100)
        elapsed = time.time() - start_time

        # ====================================================================
        # 6. RENDERING RISULTATI
        # ====================================================================
        # A) MODALITÀ MATCHING REQUISITI (Renderizza nel results_placeholder)
        if analysis_mode == "Matching Requisiti Tecnici":
            with results_placeholder.container():
                st.markdown(f"# 📋 Report Matching Requisiti")
                st.caption(f"Analisi completata in {elapsed:.1f} secondi su documenti interni.")
                st.markdown("---")

                # Contenuto della Gap Analysis
                st.markdown(result.get("executive_summary", ""))

            # Qui non renderizziamo grafici o metriche complesse

        # B) MODALITÀ STANDARD (VC DUE DILIGENCE - Renderizza nelle placeholder)
        else:
            # --- Metriche Dinamiche: Questo è il primo blocco visibile ---
            with metrics_placeholder.container():
                st.markdown(f"# 📊 {sector} Analysis Report")  # TITOLO ALL'INIZIO DEL CONTAINER
                st.caption(f"Analisi completata in {elapsed:.1f} secondi")
                st.markdown("---")

                vc_metrics = result.get("vc_metrics")
                if vc_metrics:
                    st.markdown(f"## 📊 Metriche Chiave Estratte ({sector})")

                    # --- LOGICA DI RENDERING DINAMICO ---

                    if isinstance(vc_metrics, VCMetricsProfile):
                        # --- VC PROFILO STANDARD ---
                        tab_names = ["💰 SaaS", "📈 Traction", "🌍 Market", "👥 Team", "💵 Fundraising"]
                        tab_saas, tab_traction, tab_market, tab_team, tab_fundraising = st.tabs(tab_names)

                        with tab_saas:
                            m = vc_metrics.saas_metrics
                            if m:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("ARR", f"${m.arr}M" if m.arr else "-")
                                c1.metric("Growth YoY", f"{m.revenue_growth_rate}%" if m.revenue_growth_rate else "-")
                                c2.metric("LTV/CAC", f"{m.ltv_cac_ratio}x" if m.ltv_cac_ratio else "-")
                                c2.metric("Net Retention", f"{m.net_retention_rate}%" if m.net_retention_rate else "-")
                                c3.metric("Runway", f"{m.runway_months} mo" if m.runway_months else "-")
                            else:
                                st.info("Nessuna metrica SaaS estratta.")

                        with tab_traction:
                            m = vc_metrics.traction_metrics
                            if m:
                                c1, c2 = st.columns(2)
                                c1.metric("Utenti Totali", f"{m.total_users:,}" if m.total_users else "-")
                                c2.metric("NPS Score", m.nps_score if m.nps_score else "-")
                            else:
                                st.info("Nessuna metrica Traction estratta.")

                        with tab_market:
                            m = vc_metrics.market_metrics
                            if m:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("TAM", f"${m.tam}B" if m.tam else "-")
                                c2.metric("SAM", f"${m.sam}B" if m.sam else "-")
                                c3.metric("SOM", f"${m.som}M" if m.som else "-")
                            else:
                                st.info("Nessuna metrica Market estratta.")

                        with tab_fundraising:
                            if vc_metrics.fundraising_metrics and vc_metrics.fundraising_metrics.rounds:
                                for round_data in vc_metrics.fundraising_metrics.rounds:
                                    st.markdown(f"**{round_data.stage.value}** - ${round_data.amount}M")
                                    if round_data.lead_investor: st.caption(f"Lead: {round_data.lead_investor}")
                                    st.markdown("---")
                            else:
                                st.info("Nessuna info fundraising estratta.")

                        # Rendering del team nel tab Team
                        if vc_metrics.team_metrics and vc_metrics.team_metrics.founders:
                            with tab_team:
                                for founder in vc_metrics.team_metrics.founders:
                                    st.markdown(f"**{founder.name}** - {founder.role}")
                                    if founder.background: st.caption(f"📝 {founder.background}")
                                    st.markdown("---")
                                else:
                                    st.info("Nessuna info team estratta.")

                    elif isinstance(vc_metrics, REMetricsProfile):
                        # --- REAL ESTATE PROFILO ---
                        tab_names = ["🏠 Finanziarie RE", "🌍 Mercato (VC)", "👥 Team"]
                        tab_re, tab_market, tab_team = st.tabs(tab_names)

                        with tab_re:
                            m = vc_metrics.re_metrics
                            if m:
                                st.metric("Capitalization Rate (Cap Rate)", f"{m.cap_rate}%" if m.cap_rate else "-")
                                st.metric("Internal Rate of Return (IRR)", f"{m.irr}%" if m.irr else "-")
                                st.metric("Occupancy Rate", f"{m.occupancy_rate}%" if m.occupancy_rate else "-")
                                st.metric("Net Operating Income (NOI)",
                                          f"${m.net_operating_income}M" if m.net_operating_income else "-")
                            else:
                                st.info("Nessuna metrica Real Estate estratta.")

                        # (Continuazione del rendering dinamico per gli altri settori)

                        # Rendering del team nel tab Team
                        team_metrics = getattr(vc_metrics, 'team_metrics', None)
                        if team_metrics:
                            with tab_team:
                                if team_metrics.founders:
                                    for founder in team_metrics.founders:
                                        st.markdown(f"**{founder.name}** - {founder.role}")
                                        if founder.background: st.caption(f"📝 {founder.background}")
                                        st.markdown("---")
                                else:
                                    st.info("Nessuna info team estratta.")

                        # ... (Aggiungere qui la logica per Pharma, Legal, ecc. usando lo stesso pattern with tab_nome) ...

                    # Logica aggiunta per il rendering degli altri profili metrici come Pharma, Legal...
                    # Nota: La logica dei tab Team/Fundraising deve essere adattata per Pharma/Legal se riutilizzano gli slot VC.

            # --- 2. Report Testuali (Subito dopo le Metriche) ---
            with results_placeholder.container():
                tab_summary, tab_risk, tab_feasibility, tab_facts = st.tabs([
                    "📋 Executive Summary", "🚩 Risk Analysis", "✅ Feasibility", "🔍 Fact-Check"
                ])

                with tab_summary:
                    st.markdown(result.get("executive_summary", ""))
                with tab_risk:
                    st.markdown(result.get("risk_analysis", ""))
                with tab_feasibility:
                    st.markdown(result.get("feasibility_analysis", ""))
                with tab_facts:
                    render_fact_checking_table(result.get("fact_checking_table", []))

                st.markdown("---")
                #st.markdown("### 📈 Analisi Metriche vs Benchmark")
                st.markdown(result.get("metrics_analysis", ""))

            # --- 7. GRAFO (SOLO IN STANDARD) ---
            with graph_placeholder.container():
                #st.markdown("## 🕸️ Knowledge Graph")
                if "graph_data" in result:
                    render_knowledge_graph(
                        result["graph_data"]["nodes"],
                        result["graph_data"]["edges"],
                        entity_to_analyze
                    )

        # --- 8. METADATA & DOWNLOAD (COMUNE) ---
        render_metadata_sidebar(result.get("metadata", {}))

        st.markdown("---")
        st.markdown("### 📥 Download Report")

        full_report_text = f"""# Report: {entity_to_analyze}
    Mode: {analysis_mode}

    {result.get('executive_summary', '')}

    {result.get('metrics_analysis', '')}
    {result.get('risk_analysis', '')}
    {result.get('feasibility_analysis', '')}
    """
        st.download_button("📄 Scarica Report Completo (MD)", full_report_text,
                           file_name=f"report_{entity_to_analyze}.md")

        status_container.success(f"✅ Analisi completata in {elapsed:.1f} secondi!")

    except Exception as e:
        st.error(f"❌ Errore durante l'esecuzione: {e}")
        import traceback

        with st.expander("Dettagli tecnico"):
            st.code(traceback.format_exc())
    finally:
        if agent: agent.close_graph_connection()

# Footer
st.markdown("---")
st.caption("🤖 Powered by Agentic KRAG | Built with LangChain, Neo4j, Streamlit")