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

import io
from fpdf import FPDF # Per il PDF


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
    "Ollama": [
        "llama3:8b-instruct",
        "gemma:7b-instruct",
        "mixtral:8x7b-instruct",
        "kimi-k2:1t-cloud"
    ]
}


# ============================================================================
# FUNZIONI DI UTILITÀ (ESTRAZIONE TESTO)
# ============================================================================
# INSERIRE IN: app.py (all'inizio, dopo gli import)

def inject_custom_css():
    st.markdown("""
    <style>
        /* --- UNCERTAINTY VISUALIZATION --- */
        /* Badge per confidenza */
        .metric-card {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }

        /* Confidenza Alta (Verde) */
        .conf-high {
            border-left: 5px solid #2ECC71;
        }

        /* Confidenza Bassa (Ambra/Dotted) */
        .conf-low {
            border-left: 5px solid #F39C12;
            border: 1px dashed #F39C12; /* Effetto dotted richiesto */
            background-color: #fffcf5;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2C3E50;
        }

        .metric-label {
            font-size: 14px;
            color: #7F8C8D;
            text-transform: uppercase;
        }

        .conf-badge {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            float: right;
            font-weight: bold;
        }
        .badge-high { background-color: #d4edda; color: #155724; }
        .badge-low { background-color: #fff3cd; color: #856404; }

        /* --- CONTEXT HIGHLIGHTING (SIDE PANEL) --- */
        .source-chunk {
            font-size: 12px;
            background-color: #eef;
            border-left: 3px solid #4a90e2;
            padding: 10px;
            margin-bottom: 10px;
            font-family: monospace;
        }
        .source-header {
            font-weight: bold;
            color: #4a90e2;
            margin-bottom: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

# INSERIRE IN: app.py (nella sezione Funzioni di Visualizzazione)

def render_confidence_metric(label: str, value: str, status: str = "VERIFIED", explanation: str = ""):
    """
    Renderizza una metrica con visualizzazione dell'incertezza.
    Versione FIXATA per evitare il bug del </div> visibile.
    """
    # Logica di confidenza
    # Mappiamo anche i valori italiani o nulli per sicurezza
    safe_status = str(status).upper() if status else "MISSING"

    if safe_status in ["VERIFIED", "VERIFICATA", "EXCELLENT", "GOOD"]:
        css_class = "conf-high"
        badge_class = "badge-high"
        badge_text = "HIGH CONFIDENCE"
        icon = "✅"
    else:
        # UNVERIFIED, MISSING, POOR, CONFLICTING -> Low Confidence
        css_class = "conf-low"
        badge_class = "badge-low"
        badge_text = "NEEDS VERIFICATION"
        icon = "⚠️"

    # HTML compatto senza indentazione per evitare errori di rendering
    html_code = f"""
<div class="metric-card {css_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <span class="metric-label">{label}</span>
        <span class="conf-badge {badge_class}">{icon} {badge_text}</span>
    </div>
    <div class="metric-value">{value}</div>
    <div style="font-size: 11px; color: #666; margin-top: 5px; min-height: 15px;">{explanation}</div>
</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

def render_source_inspector(source_docs: list):
    """Renderizza i chunk RAG originali nella sidebar (Ground Truth)."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Context Inspector")
        st.caption("Verifica qui i frammenti originali usati dall'AI.")

        if not source_docs:
            st.info("Nessun contesto specifico recuperato.")
            return

        for i, doc in enumerate(source_docs):
            # Estrae il nome file dai metadati o usa un default
            source_name = doc.metadata.get("source", "Documento sconosciuto")
            page_num = doc.metadata.get("page", "?")

            with st.expander(f"📄 Fonte {i + 1}: {os.path.basename(source_name)} (p.{page_num})"):
                # Evidenziazione visiva del chunk
                st.markdown(f"""
                <div class="source-chunk">
                    <div class="source-header">{source_name}</div>
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)



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


def render_requirements_sidebar(report_text: str, metadata: dict):
    """
    Renderizza una sidebar specifica per la Gap Analysis (Matching Requisiti).
    Calcola statistiche basandosi sulle emoji presenti nel report.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛠️ Gap Analysis Dashboard")

        # Parsing euristico (conta le occorrenze delle icone nel testo markdown)
        # Nota: Questo funziona perché l'LLM è istruito a usare queste emoji specifiche
        n_satisfied = report_text.count("✅")
        n_partial = report_text.count("⚠️")
        n_missing = report_text.count("❌")

        total_reqs = n_satisfied + n_partial + n_missing

        if total_reqs > 0:
            # Calcolo score di copertura (OK = 100%, Partial = 50%, Missing = 0%)
            score_pct = (n_satisfied + (0.5 * n_partial)) / total_reqs

            st.metric("Copertura Stimata", f"{score_pct:.0%}")
            st.progress(score_pct)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ OK", n_satisfied)
            with col2:
                st.metric("⚠️ Parz.", n_partial)
            with col3:
                st.metric("❌ Gap", n_missing)

        else:
            st.warning("Nessun requisito tracciato nel report.")

        st.markdown("---")
        st.markdown("### ⚡ Performance")
        st.metric("⏱️ Tempo Analisi", f"{metadata.get('analysis_time_seconds', 0):.1f}s")

        st.info("""
        **Legenda:**
        ✅ Requisito Soddisfatto
        ⚠️ Copertura Parziale
        ❌ Gap Rilevato / Non Trovato
        """)



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

    # Usiamo la palette centralizzata da config.py se disponibile, altrimenti fallback
    try:
        from config import GRAPH_COLOR_PALETTE, GRAPH_CONFIG
        color_map = GRAPH_COLOR_PALETTE
        # Creiamo un oggetto Config di agraph usando il dizionario
        # Nota: Streamlit-agraph vuole un oggetto Config, non un dict diretto
        config = Config(
            width=GRAPH_CONFIG["width"],
            height=GRAPH_CONFIG["height"],
            directed=GRAPH_CONFIG["directed"],
            physics=GRAPH_CONFIG["physics"],
            hierarchical=False,
            # Aggiungiamo opzioni extra per la UI
            interaction={"navigationButtons": True, "zoomView": True}
        )
    except ImportError:
        # Fallback se config non è importabile
        color_map = {"Startup": "#FF6B6B", "Persona": "#4ECDC4", "Investitore": "#45B7D1", "Entita": "#95E1D3"}
        config = Config(width=900, height=600, directed=True, physics=True, hierarchical=False)
    nodes = []
    seen_nodes = set()
    for n in nodes_data:
        if n["id"] in seen_nodes:
            continue
        seen_nodes.add(n["id"])
        # Determina il tipo e il colore
        node_type = n["label"].split("\n")[0] if "\n" in n["label"] else "Entita"

        # Logica speciale per il nodo centrale (Focus)
        if n["id"] == entity_name:
            color = color_map.get("focus_entity", "#FFD93D")
            size = 40  # Nodo centrale ben visibile
            font_size = 20
        else:
            color = color_map.get(node_type, "#95E1D3")
            size = 20  # Nodi periferici standard
            font_size = 14

        # Aggiungiamo il nodo con le proprietà fisiche
        nodes.append(Node(
            id=n["id"],
            label=n["label"],
            color=color,
            size=size,
            font={"size": font_size, "color": "#2C3E50"},
            # Opzionale: Shape diversa per tipo
            shape="dot"
        ))

    # Costruzione Archi
    edges = []
    for e in edges_data:
        # Colora l'arco in base al tipo di relazione (se presente nella mappa)
        # es. HA_FONDATO -> Verde, HA_INVESTITO -> Blu
        rel_type = e["label"].split(" ")[0]  # Prende la prima parola della label
        edge_color = color_map.get(rel_type, color_map.get("default_edge", "#A9D0F5"))  # Grigio default

        edges.append(Edge(
            source=e["source"],
            target=e["target"],
            label=e["label"],
            color=edge_color,
            width=2,
            # Aggiungiamo frecce per la direzione
            arrows="to"
        ))

    st.markdown(f"### 🕸️ Knowledge Graph per **{entity_name}**")
    st.caption("Usa la rotellina per zoomare e trascina per navigare.")

    # Render finale
    agraph(nodes=nodes, edges=edges, config=config)


def render_metadata_sidebar(metadata: dict):
    """Renderizza sidebar con metadata analisi (Versione Aggiornata Tesi)."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌟 Dashboard Affidabilità")

        if "error" in metadata:
            st.error(f"❌ {metadata['error']}")
            return

        # Recuperiamo il nuovo score composito
        rel_score = metadata.get('confidence_fact_check_score', 0)

        # Determina colore e label in base al punteggio
        if rel_score >= 0.7:
            score_color = "green"
            score_label = "Alta"
        elif rel_score >= 0.4:
            score_color = "orange"
            score_label = "Media"
        else:
            score_color = "red"
            score_label = "Bassa"

        st.metric("🛡️ Indice Affidabilità (Composite)", f"{rel_score:.0%}", score_label)
        st.progress(rel_score)

        st.caption(f"""
        **Composizione Score:**
        - ✅ Veridicità Fattuale (50%)
        - 🕸️ Corroborazione Grafo (30%)
        - 🎯 Confidenza Entità (20%)
        """)

        st.markdown("---")

        # Mostriamo i dettagli grezzi
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fatti Verificati", metadata.get('claims_verified', 0))
        with col2:
            st.metric("Nodi Grafo", metadata.get('graph_nodes', 0))

        st.metric("⏱️ Tempo Analisi", f"{metadata.get('analysis_time_seconds', 0):.1f}s")


# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.set_page_config(page_title="VC KRAG Analyzer", page_icon="🎯", layout="wide")

st.title("SPECTRE - Agentic Intelligence")
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
                st.markdown(f"# 🚀 {sector} Analysis Report")  # TITOLO ALL'INIZIO DEL CONTAINER
                #st.caption(f"Analisi completata in {elapsed:.1f} secondi")
                st.markdown("---")

                metrics = result.get("metrics")
                if metrics:
                    st.markdown(f"## 📈 Metriche Chiave Recuperate")

                    # --- LOGICA DI RENDERING DINAMICO ---

                    if isinstance(metrics, VCMetricsProfile):
                        # --- VC PROFILO STANDARD ---
                        tab_names = ["💰 SaaS", "📈 Traction", "🌍 Market", "👥 Team", "💵 Fundraising"]
                        tab_saas, tab_traction, tab_market, tab_team, tab_fundraising = st.tabs(tab_names)

                        with tab_saas:
                            m = metrics.saas_metrics
                            if m:
                                c1, c2, c3 = st.columns(3)
                                #c1.metric("ARR", f"${m.arr}M" if m.arr else "-")
                                #c1.metric("Growth YoY", f"{m.revenue_growth_rate}%" if m.revenue_growth_rate else "-")
                                #c2.metric("LTV/CAC", f"{m.ltv_cac_ratio}x" if m.ltv_cac_ratio else "-")
                                #c2.metric("Net Retention", f"{m.net_retention_rate}%" if m.net_retention_rate else "-")
                                #c3.metric("Runway", f"{m.runway_months} mo" if m.runway_months else "-")
                                with c1:
                                    status_arr = "MISSING"
                                    if m.arr:
                                        status_arr = m.metrics_status.get("arr", "UNVERIFIED")
                                    render_confidence_metric("ARR", f"${m.arr}M" if m.arr else "-", status_arr)

                                    status_growth = "MISSING"
                                    if m.revenue_growth_rate:
                                        status_growth = m.metrics_status.get("Growth YoY", "UNVERIFIED")
                                    render_confidence_metric("Growth YoY",
                                                             f"{m.revenue_growth_rate}%" if m.revenue_growth_rate else "-",
                                                             status_growth)
                                with c2:
                                    status_ltv = "MISSING"
                                    if m.ltv_cac_ratio:
                                        status_ltv = m.metrics_status.get("LTV/CAC", "UNVERIFIED")
                                    render_confidence_metric("LTV/CAC",
                                                             f"{m.ltv_cac_ratio}" if m.ltv_cac_ratio else "-",
                                                             status_ltv)

                                    status_retention = "MISSING"
                                    if m.net_retention_rate:
                                        status_retention = m.metrics_status.get("Net Retention", "UNVERIFIED")
                                    render_confidence_metric("Net Retention",
                                                             f"{m.net_retention_rate}" if m.net_retention_rate else "-",
                                                             status_retention)
                                with c3:
                                    status_runway = "MISSING"
                                    if m.runway_months:
                                        status_runway = m.metrics_status.get("Runway", "UNVERIFIED")
                                    render_confidence_metric("Runway",
                                                             f"{m.runway_months} Months" if m.runway_months else "-",
                                                             status_runway)
                            else:
                                st.info("Nessuna metrica SaaS estratta.")

                        with tab_traction:
                            m = metrics.traction_metrics
                            if m:
                                c1, c2 = st.columns(2)
                                #c1.metric("Utenti Totali", f"{m.total_users:,}" if m.total_users else "-")
                                #c2.metric("NPS Score", m.nps_score if m.nps_score else "-")
                                with c1:
                                    status_usr = "MISSING"
                                    if m.total_users:
                                        status_usr = m.metrics_status.get("Utenti Totali", "UNVERIFIED")
                                    render_confidence_metric("Utenti Totali", f"${m.total_users}M" if m.total_users else "-", status_usr)

                                with c2:
                                    status_nps = "MISSING"
                                    if m.nps_score:
                                        status_nps = m.metrics_status.get("NPS Score", "UNVERIFIED")
                                    render_confidence_metric("NPS Score",
                                                             f"{m.nps_score}" if m.nps_score else "-",
                                                             status_nps)
                            else:
                                st.info("Nessuna metrica Traction estratta.")

                        with tab_market:
                            m = metrics.market_metrics
                            if m:
                                c1, c2, c3 = st.columns(3)
                                #c1.metric("TAM", f"${m.tam}B" if m.tam else "-")
                                #c2.metric("SAM", f"${m.sam}B" if m.sam else "-")
                                #c3.metric("SOM", f"${m.som}M" if m.som else "-")
                                with c1:
                                    status_tam = "MISSING"
                                    if m.tam:
                                        status_tam = m.metrics_status.get("TAM", "UNVERIFIED")
                                    render_confidence_metric("TAM", f"${m.tam}M" if m.tam else "-", status_tam)

                                with c2:
                                    status_sam = "MISSING"
                                    if m.sam:
                                        status_sam = m.metrics_status.get("SAM", "UNVERIFIED")
                                    render_confidence_metric("SAM",
                                                             f"{m.sam}" if m.sam else "-",
                                                             status_sam)
                                with c3:
                                    status_som = "MISSING"
                                    if m.som:
                                        status_som = m.metrics_status.get("SOM", "UNVERIFIED")
                                    render_confidence_metric("SOM",
                                                             f"{m.som}" if m.som else "-",
                                                             status_som)
                            else:
                                st.info("Nessuna metrica Market estratta.")

                        with tab_fundraising:
                            if metrics.fundraising_metrics and metrics.fundraising_metrics.rounds:
                                for round_data in metrics.fundraising_metrics.rounds:
                                    st.markdown(f"**{round_data.stage.value}** - ${round_data.amount}M")
                                    if round_data.lead_investor: st.caption(f"Lead: {round_data.lead_investor}")
                                    st.markdown("---")
                            else:
                                st.info("Nessuna info fundraising estratta.")

                        # Rendering del team nel tab Team
                        if metrics.team_metrics and metrics.team_metrics.founders:
                            with tab_team:
                                for founder in metrics.team_metrics.founders:
                                    st.markdown(f"**{founder.name}** - {founder.role}")
                                    if founder.background: st.caption(f"📝 {founder.background}")
                                    st.markdown("---")
                                else:
                                    st.info("Nessuna info team estratta.")


                    elif isinstance(metrics, REMetricsProfile):
                        tab_names = ["🏠 Finanziarie RE", "🌍 Mercato (VC)", "👥 Team"]

                        tab_re, tab_market, tab_team = st.tabs(tab_names)

                        with tab_re:

                            m = metrics.re_metrics

                            if m:

                                # Recuperiamo lo status in modo sicuro (fallback a dizionario vuoto se manca il campo nel modello)

                                statuses = getattr(m, "metrics_status", {})

                                c1, c2 = st.columns(2)

                                with c1:

                                    render_confidence_metric("Cap Rate",

                                                             f"{m.cap_rate}%" if m.cap_rate else "-",

                                                             statuses.get("cap_rate",
                                                                          "UNVERIFIED") if m.cap_rate else "MISSING")

                                    render_confidence_metric("Occupancy Rate",

                                                             f"{m.occupancy_rate}%" if m.occupancy_rate else "-",

                                                             statuses.get("occupancy_rate",
                                                                          "UNVERIFIED") if m.occupancy_rate else "MISSING")

                                with c2:

                                    render_confidence_metric("IRR",

                                                             f"{m.irr}%" if m.irr else "-",

                                                             statuses.get("irr", "UNVERIFIED") if m.irr else "MISSING")

                                    render_confidence_metric("NOI",

                                                             f"${m.net_operating_income}M" if m.net_operating_income else "-",

                                                             statuses.get("net_operating_income",
                                                                          "UNVERIFIED") if m.net_operating_income else "MISSING")

                            else:

                                st.info("Nessuna metrica Real Estate estratta.")

                    elif isinstance(metrics, PharmaMetricsProfile):
                        # --- PHARMA & BIOTECH PROFILO ---
                        tab_names = ["🧪 R&D Pipeline", "👥 Team", "💵 Fundraising"]
                        tab_rd, tab_team, tab_fundraising = st.tabs(tab_names)

                        with tab_rd:
                            m = metrics.rd_metrics
                            if m:
                                statuses = getattr(m, "metrics_status", {})

                                # Prima riga
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    render_confidence_metric("Fase Clinica",
                                                             m.clinical_trial_phase if m.clinical_trial_phase else "-",
                                                             statuses.get("clinical_trial_phase",
                                                                          "UNVERIFIED") if m.clinical_trial_phase else "MISSING")
                                with c2:
                                    render_confidence_metric("FDA Status",
                                                             m.fda_approval_status if m.fda_approval_status else "-",
                                                             statuses.get("fda_approval_status",
                                                                          "UNVERIFIED") if m.fda_approval_status else "MISSING")
                                with c3:
                                    render_confidence_metric("Time to Market",
                                                             f"{m.time_to_market_years} anni" if m.time_to_market_years else "-",
                                                             statuses.get("time_to_market_years",
                                                                          "UNVERIFIED") if m.time_to_market_years else "MISSING")

                                st.markdown("---")

                                # Seconda riga
                                c4, c5 = st.columns(2)
                                with c4:
                                    render_confidence_metric("Scadenza Brevetto",
                                                             m.patent_expiry_date if m.patent_expiry_date else "-",
                                                             statuses.get("patent_expiry_date",
                                                                          "UNVERIFIED") if m.patent_expiry_date else "MISSING")
                                with c5:
                                    render_confidence_metric("R&D Burn Rate",
                                                             f"${m.rd_burn_rate_m}M/mo" if m.rd_burn_rate_m else "-",
                                                             statuses.get("rd_burn_rate_m",
                                                                          "UNVERIFIED") if m.rd_burn_rate_m else "MISSING")

                                if m.efficacy_data:
                                    st.info(f"📊 **Dati Efficacia**: {m.efficacy_data}")
                            else:
                                st.info("Nessuna metrica R&D estratta.")

                        # Rendering Team (Logica riutilizzata)
                        with tab_team:
                            if metrics.team_metrics and metrics.team_metrics.founders:
                                for founder in metrics.team_metrics.founders:
                                    st.markdown(f"**{founder.name}** - {founder.role}")
                                    if founder.background: st.caption(f"📝 {founder.background}")
                                    st.markdown("---")
                            else:
                                st.info("Nessuna info team estratta.")

                        # Rendering Fundraising (Logica riutilizzata)
                        with tab_fundraising:
                            if metrics.fundraising_metrics and metrics.fundraising_metrics.rounds:
                                for round_data in metrics.fundraising_metrics.rounds:
                                    st.markdown(f"**{round_data.stage.value}** - ${round_data.amount}M")
                                    if round_data.lead_investor: st.caption(f"Lead: {round_data.lead_investor}")
                                    st.markdown("---")
                            else:
                                st.info("Nessuna info fundraising estratta.")


                    elif isinstance(metrics, LegalMetricsProfile):

                        # --- LEGAL / M&A PROFILO ---

                        tab_names = ["⚖️ Risk & Compliance", "👥 Team"]

                        tab_legal, tab_team = st.tabs(tab_names)

                        with tab_legal:

                            m = metrics.legal_metrics

                            if m:

                                c1, c2 = st.columns(2)

                                # Helper per visualizzazione booleani

                                gdpr_val = "✅ Compliant" if m.gdpr_compliance else (

                                    "⚠️ Non specificato" if m.gdpr_compliance is None else "❌ Non Compliant")

                                coc_val = "✅ Presente" if m.change_of_control_clause else "❌ Assente"

                                c1.metric("GDPR / Privacy", gdpr_val)

                                c1.metric("Contenziosi Pendenti",

                                          m.pending_litigation_count if m.pending_litigation_count is not None else "0")

                                c1.metric("Status IP", m.ip_status if m.ip_status else "-")

                                c2.metric("Certificazioni (ISO/SOC)",

                                          str(m.iso_soc_certified) if m.iso_soc_certified else "-")

                                c2.metric("Clausola Change of Control", coc_val)

                            else:

                                st.info("Nessuna metrica legale estratta.")

                        # Rendering Team (Logica riutilizzata)
                        with tab_team:
                            if metrics.team_metrics and metrics.team_metrics.founders:
                                for founder in metrics.team_metrics.founders:
                                    st.markdown(f"**{founder.name}** - {founder.role}")
                                    if founder.background: st.caption(f"📝 {founder.background}")
                                    st.markdown("---")
                            else:
                                st.info("Nessuna info team estratta.")

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
        if analysis_mode == "Matching Requisiti Tecnici":
            # Mostra la dashboard tecnica personalizzata
            render_requirements_sidebar(result.get("executive_summary", ""), result.get("metadata", {}))
        else:
            # Mostra la dashboard standard VC (Confidence, Grafo, Claims)
            render_metadata_sidebar(result.get("metadata", {}))

        st.markdown("---")
        st.markdown("### 📥 Download Report")

        # 1. Preparazione Dati
        full_report_text = f"""REPORT DI ANALISI: {entity_to_analyze}
        --------------------------------------------------
        Mode: {analysis_mode}
        Data: {time.strftime("%Y-%m-%d %H:%M:%S")}

        1. EXECUTIVE SUMMARY
        --------------------
        {result.get('executive_summary', 'N/A')}

        2. RISK ANALYSIS
        ----------------
        {result.get('risk_analysis', 'N/A')}

        3. FEASIBILITY ANALYSIS
        -----------------------
        {result.get('feasibility_analysis', 'N/A')}

        4. METRICS ANALYSIS
        -------------------
        {result.get('metrics_analysis', 'N/A')}
        """

        # Preparazione DataFrame Fact-Checking per CSV/Excel
        df_facts = pd.DataFrame(result.get("fact_checking_table", []))

        # Colonne per il download (4 pulsanti in fila)
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        # --- A. DOWNLOAD TXT ---
        with col_d1:
            st.download_button(
                "📄 Report TXT",
                full_report_text,
                file_name=f"Report_{entity_to_analyze}.txt",
                mime="text/plain"
            )

        # --- B. DOWNLOAD CSV (Solo Fact Checking) ---
        with col_d2:
            if not df_facts.empty:
                csv_data = df_facts.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Fact-Check CSV",
                    csv_data,
                    file_name=f"FactCheck_{entity_to_analyze}.csv",
                    mime="text/csv"
                )
            else:
                st.button("📊 CSV (No Dati)", disabled=True)

        # --- C. DOWNLOAD EXCEL (Report Completo) ---
        with col_d3:
            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                # Foglio 1: Fact Checking
                if not df_facts.empty:
                    df_facts.to_excel(writer, sheet_name='Fact Checking', index=False)

                # Foglio 2: Summary Testuale (Hack per mettere testo in celle)
                df_summary = pd.DataFrame([x.split('\n') for x in full_report_text.split('\n\n')])
                df_summary.to_excel(writer, sheet_name='Report Text', index=False, header=False)

            st.download_button(
                "📗 Report Excel",
                buffer_excel.getvalue(),
                file_name=f"Report_{entity_to_analyze}.xlsx",
                mime="application/vnd.ms-excel"
            )

        # --- D. DOWNLOAD PDF ---
        with col_d4:
            class PDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 15)
                    self.cell(0, 10, f'VC Report: {entity_to_analyze}', 0, 1, 'C')
                    self.ln(10)


            try:
                pdf = PDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                # Sanificazione testo per FPDF (rimuove caratteri non-latin-1 che causano crash)
                safe_text = full_report_text.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, safe_text)

                pdf_output = pdf.output(dest='S').encode('latin-1')

                st.download_button(
                    "📕 Report PDF",
                    pdf_output,
                    file_name=f"Report_{entity_to_analyze}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Errore PDF: {e}")

        # --- 9. CONTEXT INSPECTOR (SIDEBAR) ---
        # Questa è la parte che mancava: attiva il pannello laterale con le fonti
        if "source_documents" in result:
            # Verifica di sicurezza se la funzione è stata incollata
            if 'render_source_inspector' in globals():
                render_source_inspector(result["source_documents"])
            else:
                st.sidebar.warning("⚠️ Funzione 'render_source_inspector' non trovata.")
        status_container.success(f"✅ Analisi completata in {elapsed:.1f} secondi!")

    except Exception as e:
        st.error(f"❌ Errore durante l'esecuzione: {e}")
        import traceback

        with st.expander("Dettagli tecnico"):
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("🤖 Powered by Agentic KRAG | Built with LangChain, Neo4j, Streamlit")