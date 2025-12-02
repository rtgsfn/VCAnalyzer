# config.py - Configurazione centralizzata per styling e comportamento

# ============================================================================
# PALETTE COLORI KNOWLEDGE GRAPH
# ============================================================================

GRAPH_COLOR_PALETTE = {
    # 1. Nodi per tipo (Entità Secondarie - Azzurro)
    "Startup": "#3498DB",  # Azzurro scuro
    "Investitore": "#3498DB",  # Azzurro scuro
    "Fondo": "#3498DB",  # Azzurro scuro
    "Entita": "#3498DB",  # Fallback azzurro scuro

    # 2. Persona (Viola)
    "Persona": "#9B59B6",  # Viola ametista

    # 3. Nodi speciali (Focus - Rosso)
    "focus_entity": "#FF4B4B",  # Rosso acceso
    "failed": "#C0392B",  # Rosso scuro (per fallimenti, se vuoi mantenerlo distinto)

    # 4. Archi / Frecce (Azzurro Pastello)
    # Cambiamo il default da azzurro pastello a nero
    "default_edge": "#000000",  # Azzurro pastello per le frecce

    # Se vuoi che TUTTE le frecce siano pastello, sovrascrivi anche le chiavi specifiche:
    # "HA_FONDATO": "#A9D0F5",
    # "HA_INVESTITO_IN": "#A9D0F5",
    # "PROBLEMI": "#A9D0F5"
    # Altrimenti, puoi commentare queste 3 righe per mantenere i colori semantici (verde/blu/rosso)
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
            "gravitationalConstant": -100,  # Aumentata repulsione (era -50)
            "centralGravity": 0.005,        # Ridotta gravità centrale (era 0.01) per allargare il grafo
            "springLength": 250,            # Aumentata lunghezza archi (era 100) per distanziare
            "springConstant": 0.05,         # Molle leggermente più morbide
            "damping": 0.4,                 # Smorzamento per evitare oscillazioni
            "avoidOverlap": 1               # Forza i nodi a non sovrapporsi
        },
        "stabilization": {
            "enabled": True,
            "iterations": 200,              # Più iterazioni per stabilizzare meglio il layout iniziale
            "fit": True
        },
        "minVelocity": 0.75
    },
    "interaction": {
        "navigationButtons": True,          # Aggiunge pulsanti zoom/pan
        "zoomView": True
    },
    # ... (restanti impostazioni di node_sizes ed edge_width invariate) ...
    "node_sizes": {
        "focus": 35,      # Leggermente più grande il nodo centrale
        "connected": 20,
        "default": 15
    },
    "edge_width": {
        "strong": 3,
        "normal": 1,      # Archi più sottili per pulizia visiva
        "weak": 1
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