import os
import json
from dotenv import load_dotenv
from typing import List, Callable, Any
import time
import concurrent.futures
import logging
from datetime import datetime

import config
# Import tool e schemi
from graph import GraphTool
from extractor import KnowledgeGraph, DocumentClaims, Claim, RelazioneFondata, RelazioneInvestimento, \
    RelazioneFallimento
from scraper import scrape_article_text

# Import componenti LangChain & API
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_tavily import TavilySearch
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from vc_metrics import (
    VCMetricsProfile, SaaSMetrics, TractionMetrics, MarketMetrics,
    TeamMetrics, FundraisingMetrics, SectorMetricsProfile, REMetricsProfile, PharmaMetricsProfile,
    LegalMetricsProfile
)

from pydantic import BaseModel

class AgenticKRAG:

    def __init__(self, provider: str, model_name: str, status_callback: Callable = None):
        """
        Inizializza l'agente KRAG.
        ...
        """
        load_dotenv()

        # Setup logging E callback PRIMA di usarli
        self._setup_logging()
        self.status_callback = status_callback

        # Ora puoi chiamare log_status in sicurezza
        self.log_status("🚀 Inizializzazione Agente KRAG", "info")

        # Inizializzazione LLM
        self.log_status(f"📡 Caricamento modello: {provider} - {model_name}", "info")
        self.llm_leggero, self.llm_pro = self._init_llm_clients(provider, model_name)

        # Inizializzazione tools
        self.log_status("🔧 Inizializzazione tools (Neo4j, Tavily, Scraper)", "info")
        self._init_tools()

        self.log_status("✅ Agente pronto per l'analisi", "success")

    def _setup_logging(self):
        """Configura il logging strutturato."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"krag_analysis_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),  # <-- AGGIUNGI QUESTO
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("NUOVA SESSIONE AGENTIC KRAG")
        self.logger.info("=" * 60)

    def log_status(self, message: str, level: str = "info"):
        """
        Log unificato: file + callback GUI.

        Args:
            message: Messaggio da loggare
            level: info, warning, error, success
        """
        # Log su file
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)

        # Callback per GUI (se presente)
        if self.status_callback:
            self.status_callback(message, level)

    def _init_llm_clients(self, provider, model_name):
        """Inizializza i client LLM (leggero + pro)."""
        api_key = ""
        if provider == "Groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key: raise ValueError("GROQ_API_KEY non trovata.")
            llm_leggero = ChatGroq(model=model_name, temperature=0, api_key=api_key, max_retries=5)
            llm_pro = ChatGroq(model=model_name, temperature=0.3, api_key=api_key, max_retries=5)
        elif provider == "Perplexity":
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key: raise ValueError("PERPLEXITY_API_KEY non trovata.")
            llm_leggero = ChatOpenAI(model=model_name, temperature=0, api_key=api_key, max_retries=5,
                                     base_url="https://api.perplexity.ai")
            llm_pro = ChatOpenAI(model=model_name, temperature=0.3, api_key=api_key, max_retries=5,
                                 base_url="https://api.perplexity.ai")
        elif provider == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key: raise ValueError("GOOGLE_API_KEY non trovata.")
            llm_leggero = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key, max_retries=5)
            llm_pro = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=api_key, max_retries=5)
        elif provider == "Ollama":
            # Default localhost, ma permettiamo override dal .env
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            # Impostiamo un timeout più alto (es. 120 secondi) perché il modello cloud
            # potrebbe metterci un po' a rispondere.
            # Nota: 'timeout' è supportato nelle versioni recenti di langchain_community
            llm_leggero = ChatOllama(model=model_name, temperature=0, base_url=base_url)
            llm_pro = ChatOllama(model=model_name, temperature=0.3, base_url=base_url)

            try:
                # Tentiamo un test rapido, ma NON blocchiamo l'app se fallisce (es. per timeout)
                print(f"Test connessione Ollama ({model_name})...")
                llm_leggero.invoke("Test")
            except Exception as e:
                # Logghiamo solo un warning invece di crashare con ConnectionError
                self.log_status(f"⚠️ Warning: Test Ollama lento o fallito ({e}), ma procedo con l'analisi.", "warning")
        else:
            raise ValueError(f"Provider '{provider}' non supportato.")
        return llm_leggero, llm_pro

    def _init_tools(self):
        """Inizializza i tools (Tavily, Neo4j)."""
        if not os.getenv("NEO4J_URI") or not os.getenv("TAVILY_API_KEY"):
            raise EnvironmentError("Assicurati che NEO4J_URI e TAVILY_API_KEY siano nel file .env")

        self.url_search_tool = TavilySearch(max_results=10, include_answer=True,
                                            tavily_api_key=os.getenv("TAVILY_API_KEY"))
        self.answer_search_tool = TavilySearch(max_results=3, include_answer=True,
                                               tavily_api_key=os.getenv("TAVILY_API_KEY"))

        self.graph_tool = GraphTool(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def close_graph_connection(self):
        """Chiude la connessione Neo4j."""
        if self.graph_tool:
            self.graph_tool.close()
            self.log_status("🔌 Connessione Neo4j chiusa", "info")

    # ============================================================================
    # NUOVA FUNZIONE: Deduzione Entità dai Documenti
    # ============================================================================

    def deduce_entity_from_document(self, document_text: str) -> dict:
        """
        Analizza il documento e deduce l'entità principale + verifica presenza.

        Returns:
            {
                "entity_found": bool,
                "entity_name": str,
                "confidence": str ("high", "medium", "low"),
                "entity_type": str ("startup", "person", "fund"),
                "context": str (breve descrizione)
            }
        """
        self.log_status("🔍 Analisi contenuto documento per deduzione entità", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un analista VC esperto. Analizza questo documento e identifica:
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
}}

ESEMPI:
Input: "Figure AI is revolutionizing robotics with humanoid robots..."
Output: {{"entity_name": "Figure AI", "entity_type": "startup", "confidence": "high", "context": "Startup di robotica umanoide"}}

Input: "Our fund, Sequoia Capital, has invested in over 200 companies..."
Output: {{"entity_name": "Sequoia Capital", "entity_type": "fund", "confidence": "high", "context": "Fondo VC con 200+ investimenti"}}"""),
            ("user", "Documento:\n\n{text}")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            result_str = chain.invoke({"text": document_text[:3000]})  # Primi 3000 char
            # Pulisci output (rimuovi markdown se presente)
            result_str = result_str.strip()
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]

            result = json.loads(result_str.strip())
            result["entity_found"] = True

            self.log_status(f"✅ Entità dedotta: {result['entity_name']} (confidenza: {result['confidence']})",
                            "success")
            return result

        except Exception as e:
            self.log_status(f"⚠️ Impossibile dedurre entità: {e}", "warning")
            return {
                "entity_found": False,
                "entity_name": "",
                "confidence": "low",
                "entity_type": "unknown",
                "context": "Analisi fallita"
            }

    # ============================================================================
    # Tool 1: RAG Semantico (OTTIMIZZATO PER VC)
    # ============================================================================

    def get_document_context(self, doc_retriever, entita: str, silent: bool = False) -> dict:
        """Recupera contesto RAG e prepara il testo con le citazioni per l'LLM."""
        if not doc_retriever:
            return {"text": "Nessun documento caricato per il RAG.", "source_documents": []}

        if not silent:
            self.log_status(f"📚 [Tool 1] Recupero contesto RAG per '{entita}'", "info")

        queries = [
            f"Qual è la strategia di business e il vantaggio competitivo di {entita}?",
            f"Chi sono i fondatori e il team chiave di {entita}? Qual è il loro background?",
            f"Quali sono i risultati finanziari, le metriche di traction e i KPI di {entita}?",
            f"Qual è il mercato target, la dimensione del mercato TAM/SAM/SOM per {entita}?"
        ]

        all_docs_objects = []
        all_contexts = []

        for i, query in enumerate(queries, 1):
            try:
                if not silent:
                    self.log_status(f"  → Query {i}/4...", "info")

                response_docs = doc_retriever.invoke(query)
                all_docs_objects.extend(response_docs)

                # --- MODIFICA CRITICA: Iniettiamo il nome del file nel testo ---
                chunk_texts = []
                for doc in response_docs:
                    # Estraiamo il nome pulito del file dai metadati
                    source_name = os.path.basename(doc.metadata.get("source", "doc_sconosciuto"))
                    # Formattiamo in modo che l'LLM lo veda chiaramente
                    chunk_texts.append(f"[[FONTE: {source_name}]]\n{doc.page_content}")

                context_text = "\n\n".join(chunk_texts)

                if context_text:
                    all_contexts.append(f"### Contesto {i}\n{context_text}")
            except Exception as e:
                if not silent:
                    self.log_status(f"  ⚠️ Errore query {i}: {e}", "warning")

        if not all_contexts:
            return {"text": "Nessun contesto trovato.", "source_documents": []}

        full_text = "\n\n".join(all_contexts)

        return {
            "text": full_text,
            "source_documents": all_docs_objects
        }

    # ============================================================================
    # Tool 2: Estrazione Affermazioni (OTTIMIZZATO PER VC)
    # ============================================================================

    def extract_claims_from_text(self, document_text: str, sector: str = "Venture Capital", silent: bool = False) -> \
        List[Claim]:
            """
            Estrae claim fattuali verificabili, adattando le priorità in base al settore.
            """
            if not silent:
                self.log_status(f"🔬 [Tool 2] Estrazione affermazioni fattuali (Focus: {sector})", "info")

            # 1. Definiamo le istruzioni specifiche per settore
            SECTOR_CLAIM_INSTRUCTIONS = {
                "Venture Capital": """
        PRIORITÀ (VC):
        1. Metriche Finanziarie: Revenue, ARR, MRR, burn rate.
        2. Traction: Clienti, growth, retention, churn.
        3. Fundraising: Round precedenti, valuation, investitori.
        4. Team: Background fondatori, exit precedenti.
        """,
                "Real Estate": """
        PRIORITÀ (Real Estate):
        1. Dati Immobile: Metratura, anno costruzione, ristrutturazioni recenti.
        2. Finanziari: NOI (Net Operating Income), Cap Rate, canoni affitto attuali.
        3. Occupancy: Tasso occupazione, durata contratti, inquilini (tenant).
        4. Location: Distanza da trasporti, sviluppi zona, claim su quartiere.
        """,
                "Pharma & Biotech": """
        PRIORITÀ (Pharma):
        1. Trial Clinici: Fase attuale (I/II/III), numero pazienti, endpoint raggiunti.
        2. Regolatorio: Status approvazione FDA/EMA, orphan drug designation.
        3. IP & Brevetti: Date scadenza brevetti, esclusività di mercato.
        4. Scientifici: Pubblicazioni citate, partnership accademiche.
        """,
                "Legal / M&A": """
        PRIORITÀ (Legal/M&A):
        1. Compliance: Certificazioni possedute (ISO, SOC2), conformità GDPR.
        2. Contenziosi: Assenza/Presenza di cause legali (litigation) passate o pendenti.
        3. Contrattuale: Clausole Change of Control, esclusività, non-compete.
        4. IP Ownership: Piena titolarità del codice/marchio, assenza violazioni.
        """
            }

            # Selezioniamo le istruzioni corrette (default VC)
            claim_hints = SECTOR_CLAIM_INSTRUCTIONS.get(sector, SECTOR_CLAIM_INSTRUCTIONS["Venture Capital"])

            try:
                extractor_chain = self.llm_pro.with_structured_output(DocumentClaims)
            except Exception as e:
                if not silent:
                    self.log_status(f"❌ Errore setup extraction chain: {e}", "error")
                return []

            # 2. Prompt Dinamico
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""Sei un analista esperto in Due Diligence per il settore **{sector}**. 
    Estrai SOLO affermazioni VERIFICABILI e CRITICHE.

    {claim_hints}

    IGNORA:
    - Opinioni soggettive ("siamo i migliori", "bellissima location")
    - Proiezioni future generiche senza basi ("diventeremo leader")
    - Marketing fluff

    ESEMPI DI OUTPUT VALIDO:
    - "La società ha un NOI di 5.2M€" (Real Estate)
    - "Il farmaco è in Fase IIb" (Pharma)
    - "Certificazione ISO 27001 ottenuta nel 2023" (Legal)
    """),
                ("user", "Documento da analizzare:\n\n{testo}")
            ])

            pipeline = prompt | extractor_chain

            try:
                # Tronchiamo a 30k char per evitare errori 400 su documenti enormi
                risultato = pipeline.invoke({"testo": document_text[:30000]})
                if not silent:
                    self.log_status(f"✅ [Tool 2] Estratte {len(risultato.claims)} affermazioni ({sector})", "success")
                return risultato.claims
            except Exception as e:
                if not silent:
                    self.log_status(f"❌ [Tool 2] Estrazione fallita: {e}", "error")
                return []
    # ============================================================================
    # Tool 3: Verifica Affermazioni
    # ============================================================================

    def verify_claims_online(self, claims: List[Claim]) -> List[dict]:
        """Verifica claim con search online."""
        self.log_status(f"🔍 [Tool 3] Avvio fact-checking per {len(claims)} affermazioni", "info")

        verified_claims_list = []
        verification_prompt = ChatPromptTemplate.from_template("""Sei un fact-checker VC. 
Data un'affermazione e delle prove dal web, determina:

VERIFICATA: Se le prove confermano chiaramente l'affermazione
FALSA: Se le prove contraddicono l'affermazione
PARZIALMENTE VERIFICATA: Se le prove confermano solo parzialmente (es. numeri diversi)
NON VERIFICABILE: Se non ci sono prove sufficienti

Rispondi con UNA SOLA delle opzioni sopra.

Affermazione: {affermazione}
Prove dal web: {prove}

Verdetto:""")

        verification_chain = verification_prompt | self.llm_leggero | StrOutputParser()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_claim = {}
            for i, claim in enumerate(claims, 1):
                query = f"{claim.soggetto} {claim.affermazione}"
                self.log_status(f"  → Claim {i}/{len(claims)}: Verifica '{claim.affermazione[:50]}...'", "info")
                future_to_claim[executor.submit(self.answer_search_tool.invoke, query)] = claim

            for future in concurrent.futures.as_completed(future_to_claim):
                claim = future_to_claim[future]
                try:
                    result_dict = future.result()
                    evidence = result_dict.get("answer", "Nessuna risposta trovata.")
                    if not evidence:
                        evidence = "Nessuna prova trovata."
                except Exception as e:
                    self.log_status(f"  ⚠️ Errore ricerca claim '{claim.affermazione[:30]}': {e}", "warning")
                    evidence = f"Ricerca fallita: {e}"

                try:
                    verdetto = verification_chain.invoke({
                        "affermazione": f"{claim.soggetto} {claim.affermazione}",
                        "prove": evidence
                    })
                    verdetto_pulito = verdetto.strip().upper()

                    # Normalizzazione output
                    if "VERIFICATA" in verdetto_pulito and "NON" not in verdetto_pulito and "PARZIALMENTE" not in verdetto_pulito:
                        verdetto_pulito = "VERIFICATA"
                    elif "PARZIALMENTE" in verdetto_pulito:
                        verdetto_pulito = "PARZIALMENTE VERIFICATA"
                    elif "FALSA" in verdetto_pulito:
                        verdetto_pulito = "FALSA"
                    else:
                        verdetto_pulito = "NON VERIFICABILE"

                except Exception as e:
                    self.log_status(f"  ⚠️ Errore verifica: {e}", "warning")
                    verdetto_pulito = "NON VERIFICABILE"

                verified_claims_list.append({
                    "soggetto": claim.soggetto,
                    "affermazione": claim.affermazione,
                    "status": verdetto_pulito,
                    "prove": evidence
                })

        # Statistiche
        stats = Counter([c["status"] for c in verified_claims_list])
        self.log_status(f"✅ [Tool 3] Fact-checking completato: {dict(stats)}", "success")

        verified_claims_list.sort(key=lambda x: [c.affermazione for c in claims].index(x['affermazione']))
        return verified_claims_list

    # ============================================================================
    # Tool 4: Graph Context
    # ============================================================================

    def get_graph_context(self, entita: str) -> str:
        """Recupera contesto dal Knowledge Graph."""
        self.log_status(f"🕸️ [Tool 4] Query Knowledge Graph per '{entita}'", "info")
        context = self.graph_tool.get_semantic_context(entita, max_hops=2)

        if "Nessun contesto trovato" in context:
            self.log_status(f"⚠️ [Tool 4] Entità '{entita}' non presente nel grafo", "warning")
        else:
            self.log_status(f"✅ [Tool 4] Contesto grafo recuperato", "success")

        return context

    # ============================================================================
    # Tool 5 & 6: Feedback Loop (invariato, solo logging aggiunto)
    # ============================================================================

    def _process_single_source(self, url: str, entita: str, silent: bool = False) -> KnowledgeGraph | None:
        """Processa una singola fonte web."""
        self.log_status(f"  🌐 Scraping: {url[:60]}...", "info")

        testo_web_sporco = scrape_article_text(url)
        if not testo_web_sporco or len(testo_web_sporco) < 50:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        prompt_pulizia = ChatPromptTemplate.from_messages([
            ("system",
             "Estrai solo il testo rilevante per '{entita}'. Se non c'è nulla di rilevante, rispondi con stringa vuota."),
            ("user", "{testo_sporco}")
        ])

        catena_pulizia = prompt_pulizia | self.llm_leggero | StrOutputParser()
        chunks = text_splitter.split_text(testo_web_sporco)
        testi_puliti = []

        for chunk in chunks:
            try:
                chunk_pulito = catena_pulizia.invoke({"entita": entita, "testo_sporco": chunk})
                testi_puliti.append(chunk_pulito)
                if "Ollama" not in str(type(self.llm_leggero)):
                    time.sleep(1)
            except Exception:
                pass

        testo_web_pulito = "\n\n".join(testi_puliti)
        if not testo_web_pulito or len(testo_web_pulito) < 20:
            return None

        knowledge_data = self._extract_knowledge_for_source(testo_web_pulito)
        return knowledge_data

    def _extract_knowledge_for_source(self, testo_pulito: str, entita_focus: str) -> KnowledgeGraph | None:
        """Estrae knowledge graph da testo."""
        try:
            extractor_chain = self.llm_pro.with_structured_output(KnowledgeGraph)
        except Exception:
            return None

        # Usiamo una f-string per inserire l'entità focus *direttamente* nel prompt
        prompt_system = f"""Sei un analista che estrae dati per un Knowledge Graph VC.

        L'entità focus di questa analisi è: **"{entita_focus}"**
        Quando estrai relazioni (FONDAZIONE, INVESTIMENTO) che coinvolgono questa entità, DEVI usare questo nome esatto.
        Non "correggerlo" o normalizzarlo. Se vedi "Cua" o "C/UA", ma il testo si riferisce a "{entita_focus}", usa "{entita_focus}".

        Estrai TUTTE le relazioni rilevanti per una due diligence VC, usando solo i seguenti tipi di relazione.

        1.  **FONDAZIONE**:
            -   `fondatore` (str): Persona che ha fondato.
            -   `azienda` (str): Azienda fondata (es. "{entita_focus}").
            -   `data_fondazione` (str): Data (YYYY-MM-DD o YYYY).

        2.  **INVESTIMENTO**:
            -   `investitore` (str): Il VC o la persona che investe.
            -   `azienda` (str): L'azienda che riceve l'investimento (es. "{entita_focus}").
            -   `importo` (float): Importo in $M (es. 50.5).
            -   `round` (str): Tipo di round (es. "Series A", "Seed").
            -   `data` (str): Data (YYYY-MM-DD o YYYY).

        3.  **FALLIMENTO**:
            -   `azienda` (str): Azienda fallita o chiusa.
            -   `data` (str): Data (YYYY-MM-DD o YYYY).

        Concentrati solo su queste tre categorie di fatti. Sii preciso con date e importi."""

        prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_system),
                ("user", "{testo_input}")
            ])


        pipeline = prompt | extractor_chain
        try:
            knowledge_data = pipeline.invoke({"testo_input": testo_pulito})
            if knowledge_data and (
                    knowledge_data.fondazioni or knowledge_data.investimenti or knowledge_data.fallimenti):
                return knowledge_data
        except Exception:
            pass
        return None

    def run_population_cycle(self, entita: str, is_deep_search: bool = False):
        """Popola il Knowledge Graph da web search (Versione Ottimizzata)."""
        self.log_status(f"🔄 [Tool 6] Avvio ciclo di popolamento per '{entita}'", "info")
        start_time = time.time()

        query_web = f"informazioni su fondatori, investimenti, partnership, competitori e storia della startup {entita}"
        try:
            search_depth = "advanced" if is_deep_search else "basic"
            max_results = 20 if is_deep_search else 10  # 10 risultati per 'normal', 20 per 'deep'

            self.log_status(f"  [Tool 6] Avvio ricerca grafo (Depth: {search_depth}, Max Results: {max_results})",
                            "info")

            search_tool = TavilySearch(
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                tavily_api_key=os.getenv("TAVILY_API_KEY")
            )
            search_result_dict = search_tool.invoke(query_web)
            search_results = search_result_dict.get("results", [])
            answer = search_result_dict.get("answer", "")
        except Exception as e:
            self.log_status(f"❌ [Tool 6] Errore web search: {e}", "error")
            return False

        if not search_results and not answer:
            self.log_status(f"⚠️ [Tool 6] Nessun risultato web trovato", "warning")
            return False

        # Usa il contenuto che Tavily ha già recuperato per noi!
        contenuti_da_processare = [result["content"] for result in search_results if result.get("content")]

        # Aggiungi la risposta riassuntiva come fonte
        if answer:
            contenuti_da_processare.append(answer)

        if not contenuti_da_processare:
            self.log_status(f"⚠️ [Tool 6] Nessun contenuto testuale estratto dalla ricerca web", "warning")
            return False

        self.log_status(f"  📄 Trovate {len(contenuti_da_processare)} fonti di testo da analizzare", "info")

        all_fondazioni, all_investimenti, all_fallimenti = [], [], []
        max_workers = 5  # Possiamo aumentare i worker perché non facciamo più scraping

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Sottomettiamo il *contenuto* E l'entità focus
            future_to_content = {executor.submit(self._extract_knowledge_for_source, content, entita): content for content in
                                 contenuti_da_processare}

            for future in concurrent.futures.as_completed(future_to_content):
                knowledge_data = future.result()
                if knowledge_data:
                    all_fondazioni.extend(knowledge_data.fondazioni)
                    all_investimenti.extend(knowledge_data.investimenti)
                    all_fallimenti.extend(knowledge_data.fallimenti)

        if not all_fondazioni and not all_investimenti and not all_fallimenti:
            self.log_status(f"⚠️ [Tool 6] Nessun dato strutturato estratto dalle fonti", "warning")
            return False

        elapsed = time.time() - start_time
        self.log_status(f"  ⏱️ Analisi completata in {elapsed:.1f}s. Avvio votazione...", "info")

        voti_fondazioni = Counter(tuple(f.model_dump().items()) for f in all_fondazioni)
        voti_investimenti = Counter(tuple(i.model_dump().items()) for i in all_investimenti)
        voti_fallimenti = Counter(tuple(fa.model_dump().items()) for fa in all_fallimenti)

        SOGLIA_CONSENSO = 1

        fatti_confermati = KnowledgeGraph(
            fondazioni=[RelazioneFondata(**dict(f)) for f, count in voti_fondazioni.items() if
                        count >= SOGLIA_CONSENSO],
            investimenti=[RelazioneInvestimento(**dict(i)) for i, count in voti_investimenti.items() if
                          count >= SOGLIA_CONSENSO],
            fallimenti=[RelazioneFallimento(**dict(fa)) for fa, count in voti_fallimenti.items() if
                        count >= SOGLIA_CONSENSO]
        )

        if not fatti_confermati.fondazioni and not fatti_confermati.investimenti and not fatti_confermati.fallimenti:
            self.log_status(f"⚠️ [Tool 6] Nessun fatto ha raggiunto consenso", "warning")
            return False

        self.log_status(
            f"  💾 Scrittura su Neo4j: {len(fatti_confermati.fondazioni)} fondazioni, {len(fatti_confermati.investimenti)} investimenti",
            "info")
        self.graph_tool.import_extracted_data(fatti_confermati)
        self.log_status(f"✅ [Tool 6] Knowledge Graph popolato con successo", "success")
        return True


    # ============================================================================
    # ORCHESTRATORE PRINCIPALE
    # ============================================================================

    def _build_fact_checking_context(self, entita: str, verified_claims: List[dict]) -> str:
        """Costruisce il contesto testuale del fact-checking."""
        if not verified_claims:
            return f"Nessuna affermazione fattuale è stata estratta dal documento su '{entita}'."

        context = f"## Document Claims Verification per '{entita}'\n\n"
        context += f"**Totale Affermazioni Analizzate**: {len(verified_claims)}\n\n"

        # Raggruppa per status
        by_status = {}
        for claim in verified_claims:
            status = claim["status"]
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(claim)

        # Ordine di priorità per il report
        status_order = ["FALSA", "PARZIALMENTE VERIFICATA", "NON VERIFICABILE", "VERIFICATA"]

        for status in status_order:
            if status in by_status:
                claims_group = by_status[status]
                icon = {"VERIFICATA": "✅", "FALSA": "❌", "PARZIALMENTE VERIFICATA": "⚠️", "NON VERIFICABILE": "❓"}[
                    status]
                context += f"\n### {icon} {status} ({len(claims_group)})\n\n"

                for claim in claims_group:
                    context += f"- **{claim['soggetto']}**: {claim['affermazione']}\n"
                    context += f"  - *Prove*: {claim['prove'][:200]}...\n\n"

        return context

    METRIC_MODEL_MAP = {
        "Venture Capital": VCMetricsProfile,
        "Real Estate": REMetricsProfile,
        "Pharma & Biotech": PharmaMetricsProfile,
        "Legal / M&A": LegalMetricsProfile,
    }

    SECTOR_METRIC_HINTS = config.SECTOR_METRIC_HINTS

    def extract_sector_metrics(self, document_text: str, entity_name: str, sector: str) -> SectorMetricsProfile:
        """
        Estrae metriche chiave specializzate dal documento.
        FIX: Usa JsonOutputParser invece di 'with_structured_output' per evitare errori di tool calling (400).
        """
        self.log_status(f"📊 Estrazione metriche chiave per '{sector}' (JSON Mode)", "info")

        # 1. Seleziona il modello Pydantic corretto
        MetricsProfileClass = self.METRIC_MODEL_MAP.get(sector, VCMetricsProfile)

        # 2. Setup del Parser JSON
        # Questo bypassa il sistema di "funzioni" dell'API che sta dando problemi
        parser = JsonOutputParser(pydantic_object=MetricsProfileClass)

        # 3. Prepara il prompt con le istruzioni di formato
        sector_hints = self.SECTOR_METRIC_HINTS.get(sector, self.SECTOR_METRIC_HINTS["Venture Capital"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un analista esperto nel settore {sector}. 
            Il tuo compito è estrarre TUTTE le metriche quantitative dal documento e valutare la confidenza del dato.

            {format_instructions}
            
            **ISTRUZIONI PER IL CAMPO 'metrics_status'**:
            Per ogni metrica estratta (es. 'arr', 'net_retention_rate'), devi popolare il dizionario 'metrics_status' con:
            - "VERIFIED": Se il dato è preciso, storico e non ambiguo (es. "Il fatturato 2023 è stato 5M").
            - "UNVERIFIED": Se il dato è una stima, una proiezione futura ("projected"), approssimativo ("circa", "more than") o ambiguo.
            - "CONFLICTING": Se ci sono valori diversi nello stesso testo.

            Esempio JSON desiderato:
            {{
              "arr": 5.0,
              "metrics_status": {{
                  "arr": "VERIFIED"
              }}
            }}

            **FOCUS SPECIFICO PER {sector_upper}**:
            {sector_hints}

            **REGOLE**:
            - Estrai solo ciò che è esplicitamente nel testo.
            - Se un valore non è presente, usa null (non inventare).
            - Rispetta rigorosamente la struttura JSON richiesta."""),
            ("user", "Entità: {entity}\n\nDocumento:\n\n{text}")
        ])

        # Pipeline: Prompt -> LLM -> Parser JSON
        pipeline = prompt | self.llm_pro | parser

        try:
            # Invocazione
            metrics_dict = pipeline.invoke({
                "entity": entity_name,
                "text": document_text[:15000],  # Troncamento sicuro
                "sector": sector,
                "sector_upper": sector.upper(),
                "sector_hints": sector_hints,
                "format_instructions": parser.get_format_instructions()
            })

            # 4. Validazione e Conversione in Oggetto Pydantic
            # Convertiamo il dizionario grezzo nell'oggetto tipizzato corretto
            metrics_obj = MetricsProfileClass(**metrics_dict)

            self.log_status(f"✅ Metriche estratte per '{entity_name}' ({MetricsProfileClass.__name__})", "success")
            return metrics_obj

        except Exception as e:
            self.log_status(f"❌ Errore estrazione metriche: {e}", "error")
            # In caso di errore, restituisci un oggetto vuoto per non bloccare il flusso
            return MetricsProfileClass(entity_name=entity_name)

    def generate_metrics_analysis(self, metrics, persona: str) -> str:
        """
        Genera analisi qualitativa delle metriche con benchmark, utilizzando la persona.
        Accetta qualsiasi profilo metrico (VC, RE, Pharma, Legal).
        """
        self.log_status(f"📈 Generazione analisi metriche vs benchmark per {type(metrics).__name__}", "info")

        # 1. Costruisci contesto metriche (dinamico)
        metrics_context = f"**METRICHE ESTRATTE PER {metrics.entity_name} ({type(metrics).__name__})**\n\n"

        # Usa la reflection di Pydantic per iterare su tutti i sub-modelli presenti
        for field_name, field_value in metrics.model_dump(exclude_none=True).items():

            # Se è un sub-modello (es. saas_metrics, re_metrics, legal_metrics)
            if isinstance(field_value, dict) and field_name not in ['metrics_status', 'entity_name']:
                # Formattazione del titolo (es. ### 💰 SaaS Metrics)
                title = field_name.replace('_', ' ').title()
                metrics_context += f"### {title}\n"

                # Itera sui singoli campi nel sub-modello
                for key, value in field_value.items():
                    if value is not None and key not in ['founders', 'rounds']:
                        metrics_context += f"- {key.replace('_', ' ').title()}: {value}\n"

                # Aggiungi liste specifiche (Founders, Rounds)
                if field_name == 'team_metrics' and 'founders' in field_value and field_value['founders']:
                    metrics_context += f"- Fondatori: {len(field_value['founders'])} membri. Esempi: {field_value['founders'][0].get('name', '')}\n"

                if field_name == 'fundraising_metrics' and 'rounds' in field_value and field_value['rounds']:
                    metrics_context += f"- Round raccolti: {len(field_value['rounds'])}.\n"

                metrics_context += "\n"

        # 2. Definisci il prompt
        # NOTA: L'LLM deve inferire i benchmark in base alla sua persona e al contesto.
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{persona}
Sei un analista che esegue un'analisi comparativa.

Compara le metriche fornite contro i BENCHMARK STANDARD del settore **{persona}**.
Se i dati sono VC-focused (ARR, LTV/CAC), usa i benchmark VC (es. Rule of 40).
Se i dati sono Real Estate (Cap Rate, Occupancy), usa i benchmark RE (es. Cap Rate > 6%).

**OUTPUT RICHIESTO** (Markdown):

## 📊 Metrics Analysis vs. Benchmarks

### ✅ Strong Metrics (Above Benchmark)
[Lista metriche che battono il benchmark con % di outperformance o commento specifico al settore. Es: "Cap Rate 7.5% vs 6.0% Benchmark."]

### ⚠️ Concern Areas (Below Benchmark)  
[Lista metriche sotto benchmark con analisi del gap e impatto sul settore.]

### ❓ Missing Critical Metrics
[Metriche chiave non presenti che servono per la valutazione (es. R&D Burn Rate per Pharma).]

### 🎯 Key Takeaways
[3-4 bullet point con conclusioni chiave, focalizzate sul settore.]

**REGOLE**:
- Sii specifico e cita i valori numerici.
- Evidenzia Red Flags (es. 'LTV/CAC 1.5x' per VC, o 'Occupancy Rate 50%' per RE).
- Se il dato è nullo ('-'), includilo in 'Missing Critical Metrics' se è vitale."""),
            ("user", "{metrics_context}")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            analysis = chain.invoke({"metrics_context": metrics_context})
            self.log_status("✅ Analisi metriche completata", "success")
            return analysis
        except Exception as e:
            self.log_status(f"❌ Errore generazione analisi metriche: {e}", "error")
            return f"## ❌ Errore Analisi Metriche\n\nSi è verificato un errore durante la generazione dell'analisi: {e}"

    def generate_risk_analysis(self, entity_name: str, fact_checking_summary: str,
                               graph_context: str, metrics_analysis: str, persona: str,
                               sector: str = "Venture Capital") -> str:
        """
        Genera analisi dei rischi specializzata per il settore.
        """
        self.log_status(f"🚩 Generazione Risk Analysis ({sector})", "info")

        if not persona:
            persona = "Sei un Partner Senior esperto in analisi dei rischi."

        # DEFINIZIONE CATEGORIE DI RISCHIO DINAMICHE
        RISK_DEFINITIONS = {
            "Venture Capital": """
    1. **Market Risk**: Timing, competizione, dimensione mercato (TAM)
    2. **Execution Risk**: Team capability, product-market fit
    3. **Financial Risk**: Burn rate, runway, unit economics
    4. **Technology Risk**: Difendibilità IP, debito tecnico
    5. **Legal/Regulatory**: Compliance, IP ownership
            """,
            "Real Estate": """
    1. **Location Risk**: Degradazione quartiere, mancanza servizi, criminalità
    2. **Tenant/Vacancy Risk**: Rischio sfitto, solvibilità inquilini, durata contratti
    3. **Structural/Capex Risk**: Manutenzione straordinaria, efficienza energetica, conformità impianti
    4. **Financial/Rate Risk**: Tassi interesse, Cap Rate compression, Cash flow volatility
    5. **Regulatory/Zoning**: Cambi destinazione d'uso, abusi edilizi
            """,
            "Pharma & Biotech": """
    1. **Clinical Risk**: Fallimento trial (Fase I/II/III), safety concerns
    2. **Regulatory Risk**: Bocciatura FDA/EMA, ritardi approvazione
    3. **Patent Cliff**: Scadenza brevetti, ingresso generici/biosimilari
    4. **Funding Risk**: Alto cash burn R&D, necessità aumenti capitale
    5. **Commercial Risk**: Pricing, rimborso assicurativo, adozione medici
            """,
            "Legal / M&A": """
    1. **Litigation Risk**: Cause pendenti, class action, passività potenziali
    2. **Compliance Risk**: Violazioni GDPR, corruzione, sicurezza sul lavoro
    3. **Contractual Risk**: Clausole Change of Control, recesso, patti non concorrenza
    4. **IP Title Risk**: Catena titolarità diritti, software open-source non dichiarato
    5. **Deal Execution**: Antitrust, approvazioni governative (Golden Power)
            """
        }

        risk_categories = RISK_DEFINITIONS.get(sector, RISK_DEFINITIONS["Venture Capital"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{persona} Stai conducendo un Risk Assessment approfondito.

    Analizza TUTTI i dati disponibili e identifica rischi categorizzati per severity.

    **CATEGORIE DI RISCHIO SPECIFICHE PER {sector.upper()} DA ANALIZZARE**:
    {risk_categories}

    **OUTPUT FORMATO**:

    ## 🚩 Risk Analysis ({sector})

    ### 🔴 CRITICAL RISKS (Deal-Breakers)
    - **[Risk Name]**: [Description + Impact + Evidence]

    ### 🟡 MEDIUM RISKS (Require Mitigation)
    - **[Risk Name]**: [Description + Mitigation Strategy]

    ### 🟢 LOW RISKS (Monitoring)
    - **[Risk Name]**: [Description]

    **REGOLE DI EXPLAINABILITY (OBBLIGATORIE)**:
    - **DEVI CITARE LE FONTI** usando questi tag alla fine di ogni punto:
        - `[DOC_RAG]`: Se trovato nel documento.
        - `[METRICS]`: Se derivato dai numeri.
        - `[FACT_CHECK]`: Se confermato/smentito dal web.
        - `[GRAPH]`: Se derivato da relazioni storiche (es. fallimenti passati).

    ESEMPIO:
    - **[Tenant Risk]**: Il contratto principale scade tra 6 mesi `[DOC_RAG]`, e il tenant è in difficoltà finanziarie secondo le news recenti `[FACT_CHECK]`.
    """),
            ("user", """Entità: {entity}

    --- Fact-Checking Summary ---
    {fact_checking}

    --- Public Knowledge Graph ---
    {graph}

    --- Metrics Analysis ---
    {metrics}

    Risk Analysis:""")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            risk_analysis = chain.invoke({
                "entity": entity_name,
                "fact_checking": fact_checking_summary,
                "graph": graph_context,
                "metrics": metrics_analysis
            })
            self.log_status("✅ Risk Analysis completata", "success")
            return risk_analysis
        except Exception as e:
            self.log_status(f"❌ Errore Risk Analysis: {e}", "error")
            return f"## ❌ Errore Risk Analysis\n\n{e}"

    def generate_feasibility_analysis(self, entity_name: str, rag_context: str,
                                      metrics_analysis: str, graph_context: str, persona: str) -> str:
        """
        Genera SOLO l'analisi di fattibilità (chiamata separata per streaming).
        """
        self.log_status("✅ Generazione Feasibility Analysis", "info")
        if not persona: persona = "Sei un Partner Senior di un fondo VC top-tier (Sequoia level)."
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{persona} Stai valutando la fattibilità di un investimento.

    Analizza la fattibilità su 3 dimensioni: Technical, Market, Financial.
    
    **OBBLIGO DI CITAZIONE**:
    Ogni volta che citi un dato tecnico o di mercato presente nei documenti, DEVI indicare la fonte tra parentesi.
    Il testo contiene tag `[[FONTE: ...]]` -> usali.
    
    **OUTPUT FORMATO**:

    ## ✅ Feasibility Analysis

    ### 🔧 Technical Feasibility
    **Team Capability**: [Score 1-5] ⭐⭐⭐⭐⭐
    [Analisi competenze tecniche basata su background team]

    **Product Stage**: [MVP/Beta/Production]
    [Assessment maturità prodotto e timeline realistica]

    **Technology Moat**: [Score 1-5] ⭐⭐⭐
    [Valutazione difendibilità tecnologica e IP]

    **Scalability**: [Assessment architettura e scaling potential]

    ### 📊 Market Feasibility  
    **Market Timing**: [Score 1-5] ⭐⭐⭐⭐
    [Early/On-Time/Late - analisi timing di mercato]

    **TAM/SAM/SOM Validation**: [Credibile/Inflated/Unrealistic]
    [Valutazione realismo dimensioni mercato dichiarate]

    **Competitive Position**: [Leader/Strong Challenger/Niche Player/Weak]
    [Posizionamento competitivo realistico]

    **Go-to-Market Strategy**: [Assessment canali acquisizione e scalabilità]

    ### 💰 Financial Feasibility
    **Unit Economics**: [Healthy/Acceptable/Concerning]
    [CAC/LTV, margini, payback period assessment]

    **Burn & Runway**: [Sustainable/Tight/Critical]
    [Analisi sostenibilità finanziaria]

    **Path to Profitability**: [Clear/Plausible/Unclear]
    [Credibilità piano finanziario e break-even timeline]

    **Capital Efficiency**: [Score 1-5] ⭐⭐⭐⭐
    [Valutazione efficienza uso capitale]

    ### 🎯 Overall Feasibility Score: [X/15] ⭐

    **REGOLE**:
    - Usa stelle per scoring (1-5)
    - Sii realistico e data-driven
    - Evidenzia gaps tra documento e realtà
    - Concludi con Overall Score su 15 (somma 3 dimensioni)"""),
            ("user", """Entità: {entity}

    --- Document Context (RAG) ---
    {rag}

    --- Metrics Analysis ---
    {metrics}

    --- Public Knowledge Graph ---
    {graph}

    Feasibility Analysis:""")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            feasibility = chain.invoke({
                "entity": entity_name,
                "rag": rag_context,
                "metrics": metrics_analysis,
                "graph": graph_context
            })
            self.log_status("✅ Feasibility Analysis completata", "success")
            return feasibility
        except Exception as e:
            self.log_status(f"❌ Errore Feasibility Analysis: {e}", "error")
            return f"## ❌ Errore Feasibility Analysis\n\n{e}"


    # ============================================================================
    # ORCHESTRATORE PRINCIPALE (Aggiornato)
    # ============================================================================
    def run_full_analysis_streaming(self, entita_focus: str, document_text: str, doc_retriever,
                                    is_deep_search: bool = False, requirements_text: str = None,
                                    sector: str = "Venture Capital") -> dict:

        # Definiamo il "Persona" in base al settore
        sector_prompts = {
            "Venture Capital": "Sei un Partner Senior di un fondo VC top-tier (Sequoia level).",
            "Real Estate": "Sei un Analista Senior di Investimenti Immobiliari.",
            "Legal / M&A": "Sei un Avvocato esperto in Due Diligence per M&A.",
            "Pharma & Biotech": "Sei un esperto di Drug Discovery e Trial Clinici."
        }

        persona = sector_prompts.get(sector, sector_prompts["Venture Capital"])

        self.log_status("=" * 60, "info")
        self.log_status(f"🎯 AVVIO ANALISI: '{entita_focus}'", "info")

        results = {}
        analysis_start = time.time()

        try:
            # ====================================================================
            # CASO 1: GAP ANALYSIS (MATCHING REQUISITI)
            # ====================================================================
            if requirements_text:
                self.log_status("📋 MODALITÀ: Gap Analysis (Solo Documenti)", "info")

                context_limit = 100000

                if len(document_text) > context_limit:
                    self.log_status("⚠️ Testo molto lungo, uso RAG + troncamento...", "warning")
                    # Strategia ibrida: Primi 50k caratteri + Ricerca RAG specifica
                    rag_query = f"Dettagli tecnici relativi a: {requirements_text[:500]}"
                    docs = doc_retriever.invoke(rag_query)
                    rag_content = "\n".join([d.page_content for d in docs])

                    context_to_use = document_text[:50000] + "\n\n--- FRAMMENTI RAG EXTRA ---\n" + rag_content
                else:
                    self.log_status("📚 Analisi su INTERO corpus documentale", "info")
                    context_to_use = document_text

                # Generazione Report
                gap_report = self.generate_requirements_match_report(
                    entita_focus,
                    requirements_text,
                    context_to_use  # Passiamo tutto il testo, non solo il RAG
                )

                results["executive_summary"] = gap_report

                # Campi vuoti per UI
                results["metrics"] = None
                results["fact_checking_table"] = []
                results["metrics_analysis"] = ""
                results["risk_analysis"] = ""
                results["feasibility_analysis"] = ""
                results["graph_data"] = {"nodes": [], "edges": []}
                results["entity_analysis"] = {"entity_name": entita_focus, "confidence": "manual"}

            # ====================================================================
            # CASO 2: STANDARD VC DUE DILIGENCE
            # ====================================================================
            else:
                self.log_status("📊 MODALITÀ: Standard VC Due Diligence", "info")
                # FASE 1: Verifica Entità
                entity_analysis = self.deduce_entity_from_document(document_text)
                results["entity_analysis"] = entity_analysis

                # FASE 2: Metriche
                self.log_status(f"📊 FASE 2: Estrazione metriche chiave per {sector}...", "info")
                metrics_from_doc = self.extract_sector_metrics(document_text, entita_focus, sector)
                metrics = self.augment_metrics_from_web(metrics_from_doc, entita_focus, is_deep_search, sector)
                results["metrics"] = metrics

                # FASE 3: RAG + Claims
                self.log_status("🔄 Analisi Documentale Parallela...", "info")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_rag = executor.submit(self.get_document_context, doc_retriever, entita_focus, silent=True)
                    future_claims = executor.submit(self.extract_claims_from_text, document_text, sector=sector,
                                                    silent=True)
                    rag_result = future_rag.result()

                    # Estrazione sicura dal dizionario
                    if isinstance(rag_result, dict):
                        contesto_rag = rag_result["text"]
                        source_docs = rag_result[
                            "source_documents"]  # Usa la chiave corretta definita in get_document_context
                    else:
                        contesto_rag = rag_result
                        source_docs = []

                    claims_list = future_claims.result()
                    results["source_documents"] = source_docs

                # FASE 4: Fact-Checking
                self.log_status("✅ Fact-Checking affermazioni...", "info")
                verified_claims = self.verify_claims_online(claims_list)
                results["fact_checking_table"] = verified_claims
                fact_checking_summary = self._build_fact_checking_context(entita_focus, verified_claims)

                # FASE 5: Grafo
                if is_deep_search:
                    self.run_population_cycle(entita_focus, is_deep_search)

                contesto_grafo = self.get_graph_context(entita_focus)
                nodes, edges = self.graph_tool.get_graph_visualization_data(entita_focus)
                results["graph_data"] = {"nodes": nodes, "edges": edges}

                # FASE 6-9: Analisi
                metrics_analysis = self.generate_metrics_analysis(metrics, persona)
                results["metrics_analysis"] = metrics_analysis

                risk_analysis = self.generate_risk_analysis(entita_focus, fact_checking_summary, contesto_grafo,
                                                            metrics_analysis, persona, sector)
                results["risk_analysis"] = risk_analysis

                feasibility_analysis = self.generate_feasibility_analysis(entita_focus, contesto_rag, metrics_analysis,
                                                                          contesto_grafo, persona)
                results["feasibility_analysis"] = feasibility_analysis

                executive_summary = self.generate_executive_summary(entita_focus, risk_analysis, feasibility_analysis,
                                                                    fact_checking_summary, persona)
                results["executive_summary"] = executive_summary

            # ====================================================================
            # CHIUSURA
            # ====================================================================
            elapsed = time.time() - analysis_start

            # 1. Fact-Checking Score (Pesato)
            # VERIFICATA = 1.0, PARZIALMENTE = 0.5, ALTRI = 0.0
            claims_list = results.get("fact_checking_table", [])
            claims_cnt = len(claims_list)
            verified_cnt = len([c for c in claims_list if c["status"] == "VERIFICATA"])
            partial_cnt = len([c for c in claims_list if c["status"] == "PARZIALMENTE VERIFICATA"])

            fact_score = 0.0
            if claims_cnt > 0:
                fact_score = (verified_cnt * 1.0 + partial_cnt * 0.5) / claims_cnt

            # 2. Graph Corroboration Score
            # Misura se esistono dati storici indipendenti.
            # Soglia: 5 nodi per avere il punteggio pieno (1.0)
            graph_nodes_cnt = len(results.get("graph_data", {}).get("nodes", []))
            graph_score = min(graph_nodes_cnt / 5, 1.0) if graph_nodes_cnt > 0 else 0.0

            # 3. Entity Confidence Score
            # Misura la sicurezza dell'LLM sull'identificazione del soggetto
            conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
            entity_conf_str = results.get("entity_analysis", {}).get("confidence", "low").lower()
            entity_score = conf_map.get(entity_conf_str, 0.2)

            # 4. CALCOLO FINALE (Media Ponderata)
            # Pesi: Fact-Check (50%), Grafo (30%), Identità (20%)
            # Questo assicura che lo score non sia mai zero se l'entità è riconosciuta.
            reliability_score = (fact_score * 0.5) + (graph_score * 0.3) + (entity_score * 0.2)

            # ====================================================================

            results["metadata"] = {
                "entity": entita_focus,
                "analysis_time_seconds": round(elapsed, 2),
                "claims_total": claims_cnt,
                "claims_verified": verified_cnt,
                "claims_partial": partial_cnt,  # Aggiunto per debug
                "graph_nodes": graph_nodes_cnt,

                "confidence_entity": results.get("entity_analysis", {}).get("confidence", "N/A"),

                # Sostituiamo il vecchio calcolo semplice con il nuovo score composito
                "confidence_fact_check_score": round(reliability_score, 2),

                # Fix per la metrica dei nodi (era forzata a 0)
                "confidence_graph_sources": graph_nodes_cnt
            }

            self.log_status(f"✅ Analisi completata in {elapsed:.1f}s", "success")
            return results

        except Exception as e:
            self.log_status(f"❌ ERRORE CRITICO: {e}", "error")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "executive_summary": f"### ❌ Errore durante l'analisi\n\n{str(e)}",
                "metadata": {"error": str(e)}
            }

    def generate_executive_summary(self, entity_name: str, risk_analysis: str,
                                   feasibility_analysis: str, fact_checking_summary: str, persona: str = "") -> str:
        """
        Genera Executive Summary finale (ultima chiamata per streaming).
        """
        self.log_status("📝 Generazione Executive Summary", "info")
        if not persona: persona = "Sei un Partner Senior di un fondo VC top-tier (Sequoia level)."
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{persona} Stai scrivendo l'Investment Summary per l'Investment Comitee.

    Sintetizza TUTTO in una raccomandazione chiara e azionabile.

    **OUTPUT FORMATO**:

    ## 📋 Executive Summary

    **Investment Thesis** (2-3 frasi):
    [Sintesi opportunità]

    **Key Strengths** (Top 3):
    1. [Punto di forza con evidenza]
    2. [Punto di forza con evidenza]  
    3. [Punto di forza con evidenza]

    **Key Concerns** (Top 3):
    1. [Concern con severity]
    2. [Concern con severity]
    3. [Concern con severity]

    **Open Questions** (Due Diligence Required):
    - [Domanda critica 1]
    - [Domanda critica 2]
    - [Domanda critica 3]

    ---

    ## 🎯 RECOMMENDATION

    **[PASS / DEEP DIVE / PASS]**

    [Giustificazione in 3-4 frasi: perché questa raccomandazione basata su risk/feasibility/opportunity]

    **Suggested Next Steps** (if DEEP DIVE):
    1. [Action item 1]
    2. [Action item 2]
    3. [Action item 3]

    ---

    **REGOLE**:
    - PASS = Investire subito (rare, high conviction)
    - DEEP DIVE = Interessante, richiede ulteriore DD
    - PASS = Non investire (risk > opportunity)
    - Raccomandazione DEVE essere supportata da dati"""),
            ("user", """Entità: {entity}

    --- Risk Analysis ---
    {risk}

    --- Feasibility Analysis ---
    {feasibility}

    --- Fact-Checking Summary ---
    {factcheck}

    Executive Summary:""")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            summary = chain.invoke({
                "entity": entity_name,
                "risk": risk_analysis,
                "feasibility": feasibility_analysis,
                "factcheck": fact_checking_summary
            })
            self.log_status("✅ Executive Summary completato", "success")
            return summary
        except Exception as e:
            self.log_status(f"❌ Errore Executive Summary: {e}", "error")
            return f"## ❌ Errore Executive Summary\n\n{e}"

    def _merge_vc_profiles(self, doc_profile: VCMetricsProfile, web_profile: VCMetricsProfile) -> VCMetricsProfile:
        """
        Fonde due profili VCMetrics, dando priorità ai dati del doc_profile.
        I dati del documento (ritenuti "source of truth") sovrascrivono i dati web.
        """
        self.log_status("  🔄 Unione profili metriche (Documento sovrascrive Web)", "info")

        # Inizia con i dati web come base
        merged = web_profile.model_copy(deep=True)

        # Aggiorna i campi di primo livello (es. business_model, current_stage)
        doc_data_top = doc_profile.model_dump(
            exclude_none=True,
            exclude={"saas_metrics", "traction_metrics", "market_metrics", "team_metrics", "fundraising_metrics"}
        )
        merged = merged.model_copy(update=doc_data_top)

        # Funzione helper per unire i sottomodelli
        def merge_submodel(doc_submodel, web_submodel, model_class):
            if not doc_submodel:
                return web_submodel

            # Se il doc ha dati ma il web no, crea un modello base
            web_base = web_submodel if web_submodel else model_class()

            # Prendi solo i campi valorizzati dal documento
            doc_data = doc_submodel.model_dump(exclude_none=True)

            # Applica l'aggiornamento
            return web_base.model_copy(update=doc_data)

        # Applica il merge per ogni sottomodello
        merged.saas_metrics = merge_submodel(doc_profile.saas_metrics, web_profile.saas_metrics, SaaSMetrics)
        merged.traction_metrics = merge_submodel(doc_profile.traction_metrics, web_profile.traction_metrics,
                                                 TractionMetrics)
        merged.market_metrics = merge_submodel(doc_profile.market_metrics, web_profile.market_metrics, MarketMetrics)
        merged.team_metrics = merge_submodel(doc_profile.team_metrics, web_profile.team_metrics, TeamMetrics)
        merged.fundraising_metrics = merge_submodel(doc_profile.fundraising_metrics, web_profile.fundraising_metrics,
                                                    FundraisingMetrics)

        # Logica speciale per le LISTE (es. fondatori, round)
        # Se il documento ha una lista, questa SOSTITUISCE quella del web
        if doc_profile.team_metrics and doc_profile.team_metrics.founders:
            if not merged.team_metrics: merged.team_metrics = TeamMetrics()
            merged.team_metrics.founders = doc_profile.team_metrics.founders

        if doc_profile.fundraising_metrics and doc_profile.fundraising_metrics.rounds:
            if not merged.fundraising_metrics: merged.fundraising_metrics = FundraisingMetrics()
            merged.fundraising_metrics.rounds = doc_profile.fundraising_metrics.rounds

        return merged

    def _merge_metrics_profiles(self, doc_profile, web_profile):
        """
        Fonde due profili metriche (Doc > Web) preservando i tipi Pydantic.
        """
        # Se i profili sono di tipo diverso, vince sempre il documento
        if type(doc_profile) != type(web_profile):
            self.log_status("⚠️ Tipi di metriche non corrispondenti. Uso dati documento.", "warning")
            return doc_profile

        # 1. Creiamo una copia base partendo dal Web (deep copy per sicurezza)
        merged = web_profile.model_copy(deep=True)

        # 2. Iteriamo sui campi del DOCUMENTO (usando model_fields per Pydantic v2)
        # Nota: Non usiamo model_dump() per evitare di convertire gli oggetti in dict
        for field_name in doc_profile.model_fields:
            if field_name == 'entity_name':
                continue

            # Preleviamo i valori reali (oggetti)
            doc_value = getattr(doc_profile, field_name)
            web_value = getattr(web_profile, field_name)

            # Se il documento non ha dati per questo campo, saltiamo (teniamo il web)
            if doc_value is None:
                continue

            # Gestione Sottomodelli (es. legal_metrics, saas_metrics)
            if isinstance(doc_value, BaseModel):
                # Se anche il web ha un valore valido per questo sottomodello, facciamo il merge
                if web_value is not None and isinstance(web_value, BaseModel):
                    web_data = web_value.model_dump(exclude_none=True)
                    doc_data = doc_value.model_dump(exclude_none=True)

                    # Merge dei dizionari: Doc sovrascrive Web
                    merged_data = {**web_data, **doc_data}

                    # RIGENERIAMO L'OGGETTO DEL TIPO CORRETTO
                    # es. LegalRiskMetrics(**merged_data)
                    new_sub_obj = type(doc_value)(**merged_data)
                    setattr(merged, field_name, new_sub_obj)
                else:
                    # Se il web è vuoto, usiamo direttamente l'oggetto del doc
                    setattr(merged, field_name, doc_value)

            # Gestione Liste e Tipi Semplici (int, str, list) -> Doc vince su tutto
            else:
                setattr(merged, field_name, doc_value)

        return merged

    def augment_metrics_from_web(self, metrics_from_doc: SectorMetricsProfile, entity_name: str, is_deep_search: bool, sector: str) -> SectorMetricsProfile:
        """
            Arricchisce il profilo metriche cercando informazioni online, guidato dal settore.
            """
        self.log_status(f"🌐 [Tool 2.5] Arricchimento metriche per '{sector}' da web", "info")

        # 1. Preparazione Query Web specifica per Settore
        sector_hints = self.SECTOR_METRIC_HINTS.get(sector, self.SECTOR_METRIC_HINTS["Venture Capital"])

        # Estraiamo le keywords chiave dal dizionario
        try:
            # Tenta di estrarre tutte le linee che contengono un ':' (per KPI chiave)
            keywords = ", ".join(
                [line.split(':')[1].strip() for line in sector_hints.split('\n') if ':' in line and len(line) > 5])
        except:
            keywords = "ARR, funding rounds, valuation, Cap Rate, Clinical Trial Phase"  # Fallback generico

        query = f"Key metrics, funding, and team for {entity_name} in {sector} focused on {keywords}"

        # 2. Ricerca Web
        try:
            search_depth = "advanced" if is_deep_search else "basic"
            max_results = 10 if is_deep_search else 5

            self.log_status(f"  [Tool 2.5] Avvio ricerca metriche (Settore: {sector}, Depth: {search_depth})", "info")

            search_tool = TavilySearch(
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                tavily_api_key=os.getenv("TAVILY_API_KEY")
            )
            search_result_dict = search_tool.invoke(query)
            search_results = search_result_dict.get("results", [])
            answer = search_result_dict.get("answer", "")
        except Exception as e:
            self.log_status(f"⚠️ [Tool 2.5] Ricerca web per metriche fallita: {e}", "warning")
            return metrics_from_doc  # Fallback: restituisce solo i dati del doc

        # 3. Estrazione Metriche dal contesto web (Usiamo lo stesso prompt dinamico)
        web_context = "\n\n".join([res['content'] for res in search_results if res.get('content')])
        if answer:
            web_context += f"\n\n--- Riepilogo Web ---\n{answer}"

        if not web_context.strip():
            self.log_status("⚠️ [Tool 2.5] Nessun contenuto web trovato per le metriche", "warning")
            return metrics_from_doc

        try:
            # Riutilizza la logica di estrazione dinamica sul contesto web
            metrics_from_web = self.extract_sector_metrics(web_context, entity_name, sector)
            self.log_status("  ✅ [Tool 2.5] Estrazione metriche da web completata", "success")
        except Exception as e:
            self.log_status(f"⚠️ [Tool 2.5] Estrazione metriche da web fallita: {e}", "warning")
            return metrics_from_doc

        # 4. Mergia i due profili
        final_metrics = self._merge_metrics_profiles(metrics_from_doc, metrics_from_web)

        return final_metrics

    def generate_requirements_match_report(self, entity_name: str, requirements_list: str,
                                           rag_context: str) -> str:
        """
        Genera un report di Gap Analysis basato ESCLUSIVAMENTE sui documenti interni.
        Include riferimenti ai file sorgente.
        """
        self.log_status("📋 Generazione Gap Analysis (Solo Documenti Interni)", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Auditor Tecnico rigoroso. 
Stai eseguendo una **Gap Analysis** per verificare se i documenti forniti soddisfano una lista di requisiti.

**FONTI DATI**:
Usa ESCLUSIVAMENTE il "Contesto Documentale" fornito. 
Il contesto contiene marcatori come `--- FILE: nome_file.pdf ---`. 
**È OBBLIGATORIO** citare il nome del file ogni volta che trovi un'evidenza.

**COMPITO**:
Per ogni requisito nella lista:
1. Cerca prove esplicite nel contesto documentale.
2. Determina lo stato: ✅ SODDISFATTO, ⚠️ PARZIALE, o ❌ NON TROVATO.
3. Cita la frase esatta e la **FONTE** (nome del file).

**OUTPUT RICHIESTO (Markdown)**:

# 🧩 Gap Analysis Record

## 📊 Sintesi
[Breve riassunto della copertura: es. "La documentazione (in particolare 'technical_specs.pdf') copre 8/10 requisiti..."]

## 📝 Dettaglio Verifica

| Requisito | Stato | Evidenza nel Documento | Fonte (File) | Note/Gap |
|-----------|-------|------------------------|--------------|----------|
| [Nome Req] | [Stato] | [Citazione Esatta] | **[Nome File]** | [Dettagli] |
...

## 🚩 Conclusioni
[Verdetto basato solo sui documenti]
"""),
            ("user", """
--- LISTA REQUISITI ---
{requirements}

--- CONTESTO DOCUMENTALE (RAG) ---
{rag}
""")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            report = chain.invoke({
                "entity": entity_name,
                "requirements": requirements_list,
                "rag": rag_context
            })
            self.log_status("✅ Gap Analysis completata", "success")
            return report
        except Exception as e:
            self.log_status(f"❌ Errore Gap Analysis: {e}", "error")
            return f"Errore durante la generazione del report: {e}"

    def generate_gap_analysis(self, entity_name: str, requirements: str, document_text: str, rag_context: str) -> str:
        """
        Agente Specializzato: Confronta Requisiti vs Documenti.
        """
        self.log_status("🕵️‍♂️ Avvio Agente Gap Analysis...", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Auditor Tecnico esperto.
Il tuo compito è eseguire una **GAP ANALYSIS** rigorosa verificando se i requisiti richiesti sono presenti nella documentazione fornita.

INPUT:
1. **LISTA REQUISITI**: Cosa cerca il cliente.
2. **DOCUMENTI FORNITI**: Pitch deck o documentazione tecnica della startup (RAG + Testo).

ISTRUZIONI:
- Per OGNI requisito nella lista, cerca prove specifiche nei documenti.
- Sii severo: se non è scritto esplicitamente, è un GAP (o "Non Specificato").
- Non inventare nulla. Basati solo sul testo fornito.

OUTPUT FORMATO (Markdown):

# 🧩 GAP ANALYSIS RECORD: {entity}

## 📊 Sintesi Copertura
[Breve riassunto: es. "Il documento copre il 70% dei requisiti tecnici..."]

## 📝 Dettaglio Verifica

| Requisito | Stato | Evidenza dal Documento | Gap / Note |
|-----------|-------|------------------------|------------|
| [Nome Req] | ✅ SODDISFATTO <br> ⚠️ PARZIALE <br> ❌ NON SODDISFATTO | [Citare esattamente la frase dal documento] | [Spiegare cosa manca] |
...

## 🚩 Conclusioni
[Verdetto finale: Procedere o richiedere integrazioni?]
"""),
            ("user", """Entità: {entity}

--- LISTA REQUISITI DA VERIFICARE ---
{requirements}

--- CONTESTO DOCUMENTALE (Estratti Rilevanti) ---
{rag}

--- TESTO DOCUMENTO (Prime parti) ---
{doc_head}
""")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            # Usiamo sia il contesto RAG specifico (se c'è) sia l'inizio del documento per sicurezza
            report = chain.invoke({
                "entity": entity_name,
                "requirements": requirements,
                "rag": rag_context,
                "doc_head": document_text[:15000]  # Passiamo un ampio chunk di testo diretto
            })
            self.log_status("✅ Gap Analysis completata", "success")
            return report
        except Exception as e:
            self.log_status(f"❌ Errore Gap Analysis: {e}", "error")
            return f"Errore durante l'analisi: {e}"

    def extract_vc_metrics(self, document_text: str, entity_name: str) -> VCMetricsProfile:
        """
        Estrae metriche VC specializzate dal documento.
        Usa output strutturato per garantire parsing affidabile.
        """
        self.log_status(f"📊 Estrazione metriche VC per '{entity_name}'", "info")

        try:
            extractor_chain = self.llm_pro.with_structured_output(VCMetricsProfile)
        except Exception as e:
            self.log_status(f"⚠️ Output strutturato non disponibile: {e}", "warning")
            return VCMetricsProfile(entity_name=entity_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un analista VC esperto nell'estrazione di metriche finanziarie.

    Estrai TUTTE le metriche quantitative dal documento. Se una metrica non è presente, lasciala come null.

    **METRICHE CRITICHE DA CERCARE**:

    1. **SaaS Metrics**:
       - ARR (Annual Recurring Revenue)
       - MRR (Monthly Recurring Revenue)  
       - Revenue Growth Rate (YoY %)
       - Gross Margin (%)
       - LTV/CAC Ratio
       - CAC Payback (months)
       - Net Retention Rate (%)
       - Gross Retention Rate (%)
       - Monthly Burn Rate
       - Runway (months)

    2. **Traction Metrics**:
       - Total Users/Customers
       - Paying Customers
       - User Growth Rate (MoM %)
       - ARPU (Average Revenue Per User)
       - NPS Score
       - Enterprise Customers (>$100K ARR)

    3. **Market Metrics**:
       - TAM (Total Addressable Market)
       - SAM (Serviceable Addressable Market)
       - SOM (Serviceable Obtainable Market)
       - Market Share (%)
       - Market Growth Rate (CAGR %)

    4. **Team**:
       - Founders (nome, ruolo, background)
       - Team Size
       - Previous Exits dei founder

    5. **Fundraising**:
       - Round precedenti (stage, amount, date, investors)
       - Total Raised
       - Last Valuation

    **REGOLE**:
    - Converti sempre i valori in formati numerici standardizzati
    - Es: "$5M ARR" → arr: 5.0
    - Es: "150% YoY growth" → revenue_growth_rate: 150.0
    - Es: "18 months runway" → runway_months: 18
    - Per team, estrai SOLO informazioni esplicite (non inventare)"""),
            ("user", "Entità: {entity}\n\nDocumento:\n\n{text}")
        ])

        pipeline = prompt | extractor_chain

        try:
            metrics = pipeline.invoke({"entity": entity_name, "text": document_text[:5000]})
            self.log_status(f"✅ Metriche estratte per '{entity_name}'", "success")
            return metrics
        except Exception as e:
            self.log_status(f"❌ Errore estrazione metriche: {e}", "error")
            return VCMetricsProfile(entity_name=entity_name)

def main():
    """Test CLI."""
    print("Questo script è pensato per essere importato da 'app.py'.")


if __name__ == "__main__":
    main()