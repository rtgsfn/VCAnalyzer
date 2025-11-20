# config.py - Configurazione centralizzata per styling e comportamento

# ============================================================================
# PALETTE COLORI KNOWLEDGE GRAPH
# ============================================================================

GRAPH_COLOR_PALETTE = {
    # Nodi per tipo
    "Startup": "#FF6B6B",  # Rosso corallo vibrante
    "Persona": "#4ECDC4",  # Turchese
    "Investitore": "#45B7D1",  # Blu chiaro
    "Fondo": "#9B59B6",  # Viola
    "Entita": "#95E1D3",  # Verde acqua (fallback)

    # Nodi speciali
    "focus_entity": "#FFD93D",  # Giallo oro per entità target
    "failed": "#E74C3C",  # Rosso scuro per fallimenti

    # Edges
    "HA_FONDATO": "#2ECC71",  # Verde per fondazioni
    "HA_INVESTITO_IN": "#3498DB",  # Blu per investimenti
    "PROBLEMI": "#E74C3C",  # Rosso per problemi/fallimenti
    "default_edge": "#7F8C8D"  # Grigio per altre relazioni
}

# ============================================================================
# CONFIGURAZIONE VISUALIZZAZIONE GRAFO
# ============================================================================

GRAPH_CONFIG = {
    "width": 900,
    "height": 600,
    "directed": True,
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
        },
        "stabilization": {
            "enabled": True,
            "iterations": 100
        }
    },
    "node_sizes": {
        "focus": 25,  # Nodo target
        "connected": 18,  # Nodi connessi
        "default": 15  # Altri nodi
    },
    "edge_width": {
        "strong": 3,  # Relazioni importanti
        "normal": 2,  # Relazioni standard
        "weak": 1  # Relazioni deboli
    }
}

# ============================================================================
# MAPPING ICONE PER FACT-CHECKING
# ============================================================================

FACT_CHECK_ICONS = {
    "VERIFICATA": "✅",
    "FALSA": "❌",
    "PARZIALMENTE VERIFICATA": "⚠️",
    "NON VERIFICABILE": "❓"
}

FACT_CHECK_COLORS = {
    "VERIFICATA": "#2ECC71",  # Verde
    "FALSA": "#E74C3C",  # Rosso
    "PARZIALMENTE VERIFICATA": "#F39C12",  # Arancione
    "NON VERIFICABILE": "#95A5A6"  # Grigio
}

# ============================================================================
# CONFIGURAZIONE LOGGING
# ============================================================================

LOG_CONFIG = {
    "log_dir": "logs",
    "log_level": "INFO",
    "log_format": "%(asctime)s | %(levelname)s | %(message)s",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# ============================================================================
# SOGLIE E PARAMETRI ANALISI
# ============================================================================

ANALYSIS_PARAMS = {
    # Consensus voting per popolamento grafo
    "consensus_threshold": 1,  # Minimo 1 fonte per accettare un fatto

    # RAG
    "rag_chunk_size": 1000,
    "rag_chunk_overlap": 200,
    "rag_top_k": 3,

    # Web scraping
    "max_sources_to_scrape": 5,
    "scrape_timeout_seconds": 10,
    "max_workers_scraping": 2,

    # Fact-checking
    "max_workers_factcheck": 5,
    "factcheck_cooldown_seconds": 1,

    # Neo4j
    "neo4j_max_hops": 2,
    "neo4j_query_limit": 20
}

# ============================================================================
# PROMPT TEMPLATES (Versioni ottimizzate)
# ============================================================================

PROMPT_TEMPLATES = {
    "entity_deduction": """Sei un analista VC esperto. Analizza questo documento e identifica:
1. L'ENTITÀ PRINCIPALE (startup, persona, o fondo VC) di cui parla il documento
2. Il TIPO di entità (startup, person, fund)
3. Il LIVELLO DI CONFIDENZA (high se molto chiaro, medium se deducibile, low se ambiguo)
4. Un BREVE CONTESTO (1 frase)

Rispondi SOLO in formato JSON:
{{
    "entity_name": "Nome Entità",
    "entity_type": "startup/person/fund",
    "confidence": "high/medium/low",
    "context": "Breve descrizione"
}}""",

    "claim_extraction": """Sei un analista VC senior. Estrai SOLO affermazioni VERIFICABILI e CRITICHE.

PRIORITÀ:
1. Metriche Finanziarie (Revenue, ARR, MRR, burn rate, runway)
2. Traction (clienti, crescita utenti, retention)
3. Fundraising (round, valuation, investitori)
4. Team (background fondatori, università, aziende precedenti)
5. Mercato (TAM/SAM/SOM, market share)
6. Prodotto (launch date, brevetti, IP)

IGNORA: Opinioni, proiezioni generiche, marketing fluff.""",

    "fact_verification": """Sei un fact-checker VC. Determina:

VERIFICATA: Prove confermano chiaramente
FALSA: Prove contraddicono
PARZIALMENTE VERIFICATA: Prove confermano parzialmente
NON VERIFICABILE: Prove insufficienti

Rispondi con UNA SOLA opzione."""
}

# ============================================================================
# MESSAGGI UI
# ============================================================================

UI_MESSAGES = {
    "welcome": "🎯 **Knowledge-Retrieval Augmented Generation** per Due Diligence Investigativa",
    "guide": """
**Step 1**: Carica documenti (pitch deck, memo, etc.)
**Step 2**: Specifica l'entità da analizzare
**Step 3**: Avvia l'analisi investigativa

Il sistema:
- ✅ Verifica claim del documento
- 🕸️ Cerca fatti pubblici
- 📊 Genera report rischi/fattibilità
""",
    "no_documents": "❌ Carica almeno un documento o incolla del testo",
    "no_entity": "❌ Specifica un'entità da analizzare",
    "analysis_complete": "✅ Analisi completata in {time:.1f} secondi!",
    "error_generic": "❌ Si è verificato un errore durante l'analisi"
}

SECTOR_METRIC_HINTS = {
        "Venture Capital": """
            METRICHE VC CRITICHE DA CERCARE:
            1. **SaaS Metrics**: ARR, MRR, Revenue Growth Rate (YoY %), LTV/CAC Ratio, Net Retention Rate (%), Runway (months), Rule of 40.
            2. **Traction Metrics**: Total Users, Paying Customers, User Growth Rate (MoM %), NPS Score.
            3. **Fundraising**: Round precedenti, Last Valuation, Total Raised.
            """,
        "Real Estate": """
            METRICHE IMMOBILIARI CRITICHE DA CERCARE:
            1. **Finanziarie**: Cap Rate (Capitalization Rate), Cash-on-Cash Return (%), ROI, IRR.
            2. **Operative**: Occupancy Rate (%), Metratura, Costi di Manutenzione, Status Permessi.
            3. **Mercato**: Zone di Mercato, Crescita della Zona (CAGR %).
            """,
        "Pharma & Biotech": """
            METRICHE R&D CRITICHE DA CERCARE:
            1. **R&D / Sviluppo**: Clinical Trial Phase (I, II, III), FDA/EMA Approval Status, Brevetti (Data Scadenza), Efficacy/Safety Data.
            2. **Finanziarie**: Burn Rate (R&D spend), Time to Market (anni), Total Raised.
            3. **Scientifiche**: Pubblicazioni Scientifiche, Key Opinion Leaders (KOLs) coinvolti.
            """,
        "Legal / M&A": """
            FATTORI DI RISCHIO LEGALE CRITICI DA CERCARE:
            1. **Compliance**: Certificazioni (ISO, SOC), GDPR compliance, Regolamentazioni di Settore.
            2. **Proprietà Intellettuale**: Status dei Brevetti (validi/pendenti), Contenziosi IP.
            3. **Contratti**: Clausole di Change of Control, Contenziosi Pendenti.
            """
    }


# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_CONFIG = {
    "report_format": "markdown",
    "graph_format": "json",
    "factcheck_format": "csv",
    "filename_pattern": "{entity}_{timestamp}_{type}"
}