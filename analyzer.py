import os
import json
from dotenv import load_dotenv
from typing import List, Callable
import time
import concurrent.futures
import logging
from datetime import datetime

# Import tool e schemi
from graph import GraphTool
from extractor import KnowledgeGraph, DocumentClaims, Claim, RelazioneFondata, RelazioneInvestimento, \
    RelazioneFallimento
from scraper import scrape_article_text

# Import componenti LangChain & API
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from vc_metrics import (
    VCMetricsProfile, SaaSMetrics, TractionMetrics, MarketMetrics,
    TeamMetrics, FundraisingMetrics, TeamMember, FundraisingRound,
    BusinessModel, FundingStage, MetricStatus,
    get_benchmark_assessment, TIER_1_VCS
)

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
            try:
                llm_leggero = ChatOllama(model=model_name, temperature=0, )
                llm_pro = ChatOllama(model=model_name, temperature=0.3)
                llm_leggero.invoke("Ciao")
            except Exception as e:
                raise ConnectionError(f"Impossibile connettersi a Ollama. Dettagli: {e}")
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

    def get_document_context(self, doc_retriever, entita: str, silent: bool = False) -> str:
        """Recupera contesto RAG ottimizzato per analisi VC."""
        if not doc_retriever:
            return "Nessun documento caricato per il RAG."

        if not silent:
            self.log_status(f"📚 [Tool 1] Recupero contesto RAG per '{entita}'", "info")

        # Query ottimizzate per VC
        queries = [
            f"Qual è la strategia di business e il vantaggio competitivo di {entita}?",
            f"Chi sono i fondatori e il team chiave di {entita}? Qual è il loro background?",
            f"Quali sono i risultati finanziari, le metriche di traction e i KPI di {entita}?",
            f"Qual è il mercato target, la dimensione del mercato TAM/SAM/SOM per {entita}?"
        ]

        all_contexts = []
        for i, query in enumerate(queries, 1):
            try:
                if not silent:
                    self.log_status(
                        f"  → Query {i}/4: Contesto {['strategico', 'team', 'finanziario', 'mercato'][i - 1]}",
                        "info")
                response_docs = doc_retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in response_docs])
                if context:
                    all_contexts.append(
                        f"### {['Strategia & Competitività', 'Team & Background', 'Metriche & Traction', 'Mercato & TAM'][i - 1]}\n{context}")
            except Exception as e:
                if not silent:
                    self.log_status(f"  ⚠️ Errore query {i}: {e}", "warning")

        if not all_contexts:
            return "Nessun contesto semantico trovato nei documenti."

        full_context = "\n\n".join(all_contexts)

        if not silent:
            self.log_status(f"✅ [Tool 1] Recuperati {len(all_contexts)} blocchi di contesto", "success")

        return full_context

    # ============================================================================
    # Tool 2: Estrazione Affermazioni (OTTIMIZZATO PER VC)
    # ============================================================================

    def extract_claims_from_text(self, document_text: str, silent: bool = False) -> List[Claim]:
            """Estrae claim focalizzati su metriche VC critiche."""
            if not silent:
                self.log_status("🔬 [Tool 2] Estrazione affermazioni fattuali (focus VC)", "info")

            try:
                extractor_chain = self.llm_pro.with_structured_output(DocumentClaims)
            except Exception as e:
                if not silent:
                    self.log_status(f"❌ Errore setup extraction chain: {e}", "error")
                return []

            prompt = ChatPromptTemplate.from_messages([
                ("system", """Sei un analista VC senior. Estrai SOLO affermazioni VERIFICABILI e CRITICHE per una due diligence.

    PRIORITÀ (ordine di importanza):
    1. **Metriche Finanziarie**: Revenue, ARR, MRR, burn rate, runway, profitability
    2. **Traction**: Numero clienti, user growth, retention rate, NPS
    3. **Fundraising**: Round precedenti, valuation, investitori, termini
    4. **Team**: Background fondatori (aziende precedenti, università, exit)
    5. **Mercato**: Market size (TAM/SAM/SOM), competitori, market share
    6. **Prodotto**: Launch date, feature specifiche, brevetti, IP

    IGNORA:
    - Opinioni soggettive ("siamo i migliori", "rivoluzionario")
    - Proiezioni future generiche ("cresceremo 10x")
    - Marketing fluff senza numeri

    ESEMPI VC-FOCUSED:

    [INPUT 1]
    "La nostra startup ha raggiunto $2M ARR nel 2023 con una crescita MoM del 15%. Abbiamo 50 clienti enterprise e un net retention rate del 120%."

    [OUTPUT 1]
    [
        {{"soggetto": "[Nome Startup]", "affermazione": "ha raggiunto $2M ARR nel 2023"}},
        {{"soggetto": "[Nome Startup]", "affermazione": "ha una crescita MoM del 15%"}},
        {{"soggetto": "[Nome Startup]", "affermazione": "ha 50 clienti enterprise"}},
        {{"soggetto": "[Nome Startup]", "affermazione": "ha un net retention rate del 120%"}}
    ]

    [INPUT 2]
    "Il nostro CEO, Mario Rossi, è ex-VP Engineering di Google e ha un PhD in AI da Stanford. Il nostro CTO ha fondato e venduto due startup SaaS."

    [OUTPUT 2]
    [
        {{"soggetto": "Mario Rossi", "affermazione": "è stato VP Engineering di Google"}},
        {{"soggetto": "Mario Rossi", "affermazione": "ha un PhD in AI da Stanford"}},
        {{"soggetto": "[CTO Nome]", "affermazione": "ha fondato e venduto due startup SaaS"}}
    ]

    [INPUT 3]
    "Il mercato del nostro settore crescerà rapidamente. Siamo la soluzione migliore."

    [OUTPUT 3]
    []
    """),
                ("user", "Documento da analizzare:\n\n{testo}")
            ])

            pipeline = prompt | extractor_chain

            try:
                risultato = pipeline.invoke({"testo": document_text})
                if not silent:
                    self.log_status(f"✅ [Tool 2] Estratte {len(risultato.claims)} affermazioni verificabili", "success")
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
    # Tool 7: Sintesi Finale (OTTIMIZZATO PER VC CON RISK/FEASIBILITY)
    # ============================================================================

    def genera_analisi_combinata(self, domanda_focus: str, contesto_grafo: str,
                                 contesto_fact_checking: str, contesto_rag_semantico: str) -> str:
        """Genera report VC strutturato con analisi rischi e fattibilità."""
        self.log_status("📊 [Tool 7] Generazione report investigativo finale", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Partner Senior di un fondo VC top-tier (Sequoia, a16z level). 
Stai preparando un Investment Committee Memo per decidere se investire in questa opportunità.

Il tuo compito è integrare TRE fonti di dati:
1. **Document Claims (Fact-Checking)**: Verifica delle affermazioni nel pitch deck
2. **Document Context (RAG)**: Contesto qualitativo estratto dal documento privato
3. **Public Knowledge Graph**: Dati pubblici verificati (fondatori, investimenti, storia)

# OUTPUT RICHIESTO (FORMATO OBBLIGATORIO)

## 📋 Executive Summary
[2-3 frasi: Tesi di investimento sintetica + raccomandazione preliminare]

## ✅ Investment Thesis (Bandiere Verdi)
[Elenca 3-5 punti di forza dell'opportunità, supportati da EVIDENZE dalle 3 fonti]
- **[Punto di Forza]**: [Evidenza concreta con fonte]

## 🚩 Risk Analysis (Bandiere Rosse)
[Analisi critica dei rischi, suddivisi per categoria]

### 🔴 CRITICAL RISKS (Deal-Breaker Potential)
- **[Rischio]**: [Descrizione + gravità + evidenza]

### 🟡 MEDIUM RISKS (Require Mitigation)
- **[Rischio]**: [Descrizione + mitigazione suggerita]

### 🟢 LOW RISKS (Monitoring Required)
- **[Rischio]**: [Descrizione]

## 📊 Feasibility Analysis
[Valutazione della fattibilità tecnica, di mercato, finanziaria]

### Technical Feasibility
- **Team Capability**: [Valutazione competenze tecniche del team basata su background]
- **Product Development Stage**: [MVP/Beta/Production e timeline realistica]
- **Technology Moat**: [Difendibilità tecnologica e barriere all'entrata]

### Market Feasibility
- **Market Timing**: [Il mercato è pronto? Early/On-time/Late]
- **TAM/SAM/SOM Analysis**: [Validazione dimensioni mercato dichiarate]
- **Competitive Position**: [Differenziazione vs competitori esistenti]

### Financial Feasibility
- **Unit Economics**: [CAC/LTV ratio, margini, payback period]
- **Burn Rate & Runway**: [Sostenibilità finanziaria]
- **Path to Profitability**: [Credibilità del piano finanziario]

## ⚠️ Open Questions (Due Diligence Necessaria)
[Domande critiche rimaste senza risposta che richiedono ulteriore investigazione]
1. [Domanda specifica da investigare]

## 🎯 Recommendation
[PASS / DEEP DIVE / PASS (with rationale)]
[Giustificazione della raccomandazione in 3-4 frasi]

---

**REGOLE CRITICHE**:
- Ogni affermazione DEVE essere supportata da evidenze dalle 3 fonti
- Se documento e fatti pubblici divergono, EVIDENZIA LA DISCREPANZA con emoji 🚨
- Se un claim è "FALSO" nel fact-checking, è AUTOMATICAMENTE un RED FLAG 🔴
- Se dati critici (revenue, team background) sono "NON VERIFICABILI", è un YELLOW FLAG 🟡
- Sii diretto e specifico: questo è per un Investment Committee, non per marketing"""),
            ("user", """Entità Focus: {domanda_focus}

--- FONTE 1: Document Claims Verification (Fact-Checking) ---
{contesto_fact_checking}

--- FONTE 2: Document Semantic Context (RAG) ---
{contesto_rag_semantico}

--- FONTE 3: Public Knowledge Graph (Verified Facts) ---
{contesto_grafo}

Investment Committee Memo:""")
        ])

        catena_analisi = prompt | self.llm_pro | StrOutputParser()

        try:
            analisi = catena_analisi.invoke({
                "domanda_focus": domanda_focus,
                "contesto_grafo": contesto_grafo,
                "contesto_fact_checking": contesto_fact_checking,
                "contesto_rag_semantico": contesto_rag_semantico
            })
            self.log_status("✅ [Tool 7] Report generato con successo", "success")
            return analisi
        except Exception as e:
            self.log_status(f"❌ [Tool 7] Errore generazione report: {e}", "error")
            return f"Errore nella generazione del report: {e}"

    # ============================================================================
    # ORCHESTRATORE PRINCIPALE
    # ============================================================================

    def run_full_analysis(self, entita_focus: str, document_text: str, doc_retriever) -> dict:
        """
        Esegue l'analisi completa e restituisce risultati strutturati.

        Returns:
            {
                "report": str (markdown report),
                "fact_checking_table": List[dict],
                "graph_data": {"nodes": [...], "edges": [...]},
                "metadata": {...}
            }
        """
        self.log_status("=" * 60, "info")
        self.log_status(f"🎯 INIZIO ANALISI INVESTIGATIVA: '{entita_focus}'", "info")
        self.log_status("=" * 60, "info")

        analysis_start = time.time()

        try:
            # FASE 1: Verifica presenza entità nel documento
            self.log_status("🔍 FASE 1: Verifica presenza entità nel documento", "info")
            entity_analysis = self.deduce_entity_from_document(document_text)

            # Se l'entità richiesta non corrisponde a quella dedotta
            if entity_analysis["entity_found"] and entity_analysis["entity_name"].lower() != entita_focus.lower():
                self.log_status(
                    f"⚠️ ATTENZIONE: Entità richiesta ('{entita_focus}') diversa da quella nel documento ('{entity_analysis['entity_name']}')",
                    "warning")

            # FASE 2: Elaborazione parallela (RAG + Claims)
            self.log_status("🔄 FASE 2: Elaborazione parallela (RAG + Estrazione Claims)", "info")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_rag = executor.submit(self.get_document_context, doc_retriever, entita_focus)
                future_claims = executor.submit(self.extract_claims_from_text, document_text)

                contesto_rag_semantico = future_rag.result()
                claims_list = future_claims.result()

            # FASE 3: Fact-Checking
            self.log_status("🔍 FASE 3: Fact-Checking delle affermazioni", "info")
            verified_claims = self.verify_claims_online(claims_list)

            # Costruisci tabella fact-checking
            contesto_fact_checking = self._build_fact_checking_context(entita_focus, verified_claims)

            # FASE 4: Knowledge Graph
            self.log_status("🕸️ FASE 4: Interrogazione Knowledge Graph", "info")
            contesto_grafo = self.get_graph_context(entita_focus)

            # FASE 5: Feedback Loop (se necessario)
            report_feedback = ""
            if "Nessun contesto trovato" in contesto_grafo:
                self.log_status(f"⚠️ FASE 5: Entità '{entita_focus}' non nel grafo. Avvio popolamento automatico...",
                                "warning")
                successo_popolamento = self.run_population_cycle(entita_focus)

                if successo_popolamento:
                    self.log_status("✅ Grafo popolato. Recupero contesto aggiornato...", "success")
                    contesto_grafo = self.get_graph_context(entita_focus)
                    report_feedback = f"*(Nota: l'entità '{entita_focus}' è stata aggiunta automaticamente al Knowledge Graph tramite ricerca web)*\n\n"
                else:
                    self.log_status("❌ Popolamento automatico fallito", "error")
                    contesto_grafo = f"Impossibile recuperare informazioni pubbliche per '{entita_focus}'."

            # FASE 6: Recupero dati visualizzazione grafo
            self.log_status("📊 FASE 6: Preparazione visualizzazione grafo", "info")
            nodes, edges = self.graph_tool.get_graph_visualization_data(entita_focus)

            # FASE 7: Sintesi Finale
            self.log_status("📝 FASE 7: Generazione Investment Committee Memo", "info")
            analisi_finale = self.genera_analisi_combinata(
                entita_focus,
                contesto_grafo,
                contesto_fact_checking,
                contesto_rag_semantico
            )

            # Calcola statistiche finali
            elapsed = time.time() - analysis_start
            self.log_status("=" * 60, "info")
            self.log_status(f"✅ ANALISI COMPLETATA IN {elapsed:.1f} SECONDI", "success")
            self.log_status("=" * 60, "info")

            return {
                "report": report_feedback + analisi_finale,
                "fact_checking_table": verified_claims,
                "graph_data": {"nodes": nodes, "edges": edges},
                "entity_analysis": entity_analysis,
                "metadata": {
                    "entity": entita_focus,
                    "claims_total": len(claims_list),
                    "claims_verified": len([c for c in verified_claims if c["status"] == "VERIFICATA"]),
                    "claims_false": len([c for c in verified_claims if c["status"] == "FALSA"]),
                    "claims_partial": len([c for c in verified_claims if c["status"] == "PARZIALMENTE VERIFICATA"]),
                    "claims_unverifiable": len([c for c in verified_claims if c["status"] == "NON VERIFICABILE"]),
                    "graph_nodes": len(nodes),
                    "graph_edges": len(edges),
                    "analysis_time_seconds": round(elapsed, 2)
                }
            }

        except Exception as e:
            self.log_status(f"❌ ERRORE CRITICO: {e}", "error")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "report": f"# ❌ Analisi Fallita\n\nErrore critico: {e}",
                "fact_checking_table": [],
                "graph_data": {"nodes": [], "edges": []},
                "entity_analysis": {"entity_found": False},
                "metadata": {"error": str(e)}
            }

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

    # Aggiungi questo import all'inizio del file analyzer.py esistente
    from vc_metrics import (
        VCMetricsProfile, SaaSMetrics, TractionMetrics, MarketMetrics,
        TeamMetrics, FundraisingMetrics, TeamMember, FundraisingRound,
        BusinessModel, FundingStage, MetricStatus,
        get_benchmark_assessment, TIER_1_VCS
    )

    # Aggiungi questa nuova funzione nella classe AgenticKRAG

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

    def generate_metrics_analysis(self, metrics: VCMetricsProfile) -> str:
        """
        Genera analisi qualitativa delle metriche con benchmark.
        QUESTA È UNA CHIAMATA SEPARATA per streaming.
        """
        self.log_status("📈 Generazione analisi metriche vs benchmark", "info")

        # Costruisci contesto metriche
        metrics_context = f"""
    **METRICHE ESTRATTE PER {metrics.entity_name}**

    """

        if metrics.saas_metrics:
            m = metrics.saas_metrics
            metrics_context += f"""
    ### 💰 SaaS Metrics
    - ARR: ${m.arr}M (se disponibile)
    - MRR: ${m.mrr}K
    - Revenue Growth: {m.revenue_growth_rate}% YoY
    - Gross Margin: {m.gross_margin}%
    - LTV/CAC: {m.ltv_cac_ratio}
    - CAC Payback: {m.cac_payback_months} months
    - Net Retention: {m.net_retention_rate}%
    - Monthly Burn: ${m.monthly_burn}K
    - Runway: {m.runway_months} months
    - Rule of 40: {m.rule_of_40}
    """

        if metrics.traction_metrics:
            m = metrics.traction_metrics
            metrics_context += f"""
    ### 📈 Traction Metrics
    - Total Users: {m.total_users}
    - Paying Customers: {m.paying_customers}
    - User Growth: {m.user_growth_rate}% MoM
    - ARPU: ${m.revenue_per_customer}
    - NPS: {m.nps_score}
    - Enterprise Customers: {m.enterprise_customers}
    """

        if metrics.market_metrics:
            m = metrics.market_metrics
            metrics_context += f"""
    ### 🌍 Market Metrics
    - TAM: ${m.tam}B
    - SAM: ${m.sam}B
    - SOM: ${m.som}M
    - Market Share: {m.market_share}%
    - Market Growth: {m.market_growth_rate}% CAGR
    """

        if metrics.fundraising_metrics and metrics.fundraising_metrics.rounds:
            metrics_context += f"""
    ### 💵 Fundraising History
    - Total Raised: ${metrics.fundraising_metrics.total_raised}M
    - Last Valuation: ${metrics.fundraising_metrics.last_valuation}M
    - Rounds: {len(metrics.fundraising_metrics.rounds)}
    """

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Partner Senior VC che analizza metriche finanziarie.

    Compara le metriche fornite contro i BENCHMARK STANDARD del settore VC:

    **BENCHMARK SAAS (Series A)**:
    - ARR Growth: >200% = Excellent, >150% = Good, >100% = Acceptable
    - Net Retention: >120% = Excellent, >110% = Good, >100% = Acceptable
    - LTV/CAC: >4 = Excellent, >3 = Good, >2 = Acceptable
    - CAC Payback: <6 months = Excellent, <12 = Good, <18 = Acceptable
    - Gross Margin: >80% = Excellent, >70% = Good, >60% = Acceptable
    - Rule of 40: >40 = Excellent, 20-40 = Good, <20 = Poor

    **OUTPUT RICHIESTO** (Markdown):

    ## 📊 Metrics Analysis vs. Benchmarks

    ### ✅ Strong Metrics (Above Benchmark)
    [Lista metriche che battono il benchmark con % di outperformance]

    ### ⚠️ Concern Areas (Below Benchmark)  
    [Lista metriche sotto benchmark con analisi del gap]

    ### ❓ Missing Critical Metrics
    [Metriche chiave non presenti nel documento che servono per valutazione]

    ### 🎯 Key Takeaways
    [3-4 bullet point con conclusioni chiave]

    **REGOLE**:
    - Sii specifico: "ARR Growth 250% vs 200% benchmark (+50pp)" 
    - Evidenzia red flags: "CAC Payback 24 months (2x benchmark acceptable)"
    - Se mancano metriche critiche, evidenzialo come risk"""),
            ("user", "{metrics_context}")
        ])

        chain = prompt | self.llm_pro | StrOutputParser()

        try:
            analysis = chain.invoke({"metrics_context": metrics_context})
            self.log_status("✅ Analisi metriche completata", "success")
            return analysis
        except Exception as e:
            self.log_status(f"❌ Errore generazione analisi metriche: {e}", "error")
            return f"## ❌ Errore Analisi Metriche\n\n{e}"

    def generate_risk_analysis(self, entity_name: str, fact_checking_summary: str,
                               graph_context: str, metrics_analysis: str) -> str:
        """
        Genera SOLO l'analisi dei rischi (chiamata separata per streaming).
        """
        self.log_status("🚩 Generazione Risk Analysis", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Partner VC che conduce risk assessment.

    Analizza TUTTI i dati disponibili e identifica rischi investimento, categorizzati per severity.

    **OUTPUT FORMATO**:

    ## 🚩 Risk Analysis

    ### 🔴 CRITICAL RISKS (Deal-Breakers)
    [Rischi che potrebbero far passare sull'investment]
    - **[Risk Name]**: [Description + Impact + Evidence]

    ### 🟡 MEDIUM RISKS (Require Mitigation)
    [Rischi gestibili ma che richiedono piano di mitigazione]
    - **[Risk Name]**: [Description + Mitigation Strategy]

    ### 🟢 LOW RISKS (Monitoring)
    [Rischi minori da monitorare]

    **CATEGORIE DI RISCHIO DA ANALIZZARE**:
    1. **Market Risk**: Timing, competizione, dimensione mercato
    2. **Execution Risk**: Team capability, product-market fit
    3. **Financial Risk**: Burn rate, unit economics, capitalization
    4. **Technology Risk**: Moat difendibilità, scalabilità tecnica
    5. **Team Risk**: Founder experience, key person dependency
    6. **Legal/Regulatory Risk**: Compliance, IP, litigation

    **REGOLE**:
    - CRITICAL = può far fallire l'investment
    - MEDIUM = richiede azione ma gestibile
    - LOW = awareness sufficiente
    - EVIDENZIA discrepanze tra documento e fatti pubblici come RED FLAG
    - **DEVI ASSOLUTAMENTE CITARE LE TUE FONTI PER OGNI PUNTO.**
    - Usa questi tag di citazione alla fine di ogni frase o punto:
        - `[DOC_RAG]`: Se l'informazione proviene dal contesto semantico del documento.
        - `[METRICS]`: Se l'informazione proviene dall'analisi delle metriche VC.
        - `[FACT_CHECK]`: Se l'informazione proviene dal fact-checking (es. claim FALSO o VERIFICATO).
        - `[GRAPH]`: Se l'informazione proviene dal Knowledge Graph pubblico.
        - `[DOC_METRICS]`: Se l'informazione proviene dalle metriche estratte DAL SOLO documento.
        
    ESEMPIO:
    - **[Rischio]**: Il team dichiara esperienza decennale `[DOC_RAG]`, ma questa affermazione è risultata NON VERIFICABILE `[FACT_CHECK]`.
    - **[Rischio]**: L'ARR è inferiore ai benchmark di settore `[METRICS]`.
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
                                      metrics_analysis: str, graph_context: str) -> str:
        """
        Genera SOLO l'analisi di fattibilità (chiamata separata per streaming).
        """
        self.log_status("✅ Generazione Feasibility Analysis", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Partner VC che valuta la fattibilità di un investment.

    Analizza la fattibilità su 3 dimensioni: Technical, Market, Financial.

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
                                    is_deep_search: bool = False, requirements_text: str = None) -> dict:
        """
        Esegue l'analisi completa.
        """
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
                results["vc_metrics"] = None
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
                self.log_status("📊 Estrazione metriche VC...", "info")
                metrics_from_doc = self.extract_vc_metrics(document_text, entita_focus)
                vc_metrics = self.augment_vc_metrics_from_web(metrics_from_doc, entita_focus, is_deep_search)
                results["vc_metrics"] = vc_metrics

                # FASE 3: RAG + Claims
                self.log_status("🔄 Analisi Documentale Parallela...", "info")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_rag = executor.submit(self.get_document_context, doc_retriever, entita_focus, silent=True)
                    future_claims = executor.submit(self.extract_claims_from_text, document_text, silent=True)

                    contesto_rag = future_rag.result()
                    claims_list = future_claims.result()

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
                metrics_analysis = self.generate_metrics_analysis(vc_metrics)
                results["metrics_analysis"] = metrics_analysis

                risk_analysis = self.generate_risk_analysis(entita_focus, fact_checking_summary, contesto_grafo,
                                                            metrics_analysis)
                results["risk_analysis"] = risk_analysis

                feasibility_analysis = self.generate_feasibility_analysis(entita_focus, contesto_rag, metrics_analysis,
                                                                          contesto_grafo)
                results["feasibility_analysis"] = feasibility_analysis

                executive_summary = self.generate_executive_summary(entita_focus, risk_analysis, feasibility_analysis,
                                                                    fact_checking_summary)
                results["executive_summary"] = executive_summary

            # ====================================================================
            # CHIUSURA
            # ====================================================================
            elapsed = time.time() - analysis_start

            claims_cnt = len(results.get("fact_checking_table", []))
            verified_cnt = len([c for c in results.get("fact_checking_table", []) if c["status"] == "VERIFICATA"])
            graph_nodes_cnt = len(results.get("graph_data", {}).get("nodes", []))

            results["metadata"] = {
                "entity": entita_focus,
                "analysis_time_seconds": round(elapsed, 2),
                "claims_total": claims_cnt,
                "claims_verified": verified_cnt,
                "graph_nodes": graph_nodes_cnt,
                "confidence_entity": results.get("entity_analysis", {}).get("confidence", "N/A"),
                "confidence_fact_check_score": verified_cnt / claims_cnt if claims_cnt > 0 else 0,
                "confidence_graph_sources": 0
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
                                   feasibility_analysis: str, fact_checking_summary: str) -> str:
        """
        Genera Executive Summary finale (ultima chiamata per streaming).
        """
        self.log_status("📝 Generazione Executive Summary", "info")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un Partner VC che scrive l'Executive Summary per l'Investment Committee.

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

    def augment_vc_metrics_from_web(self, metrics_from_doc: VCMetricsProfile, entity_name: str, is_deep_search: bool = False) -> VCMetricsProfile:
        """
        Arricchisce il profilo metriche VC cercando informazioni online.
        I dati trovati nel documento (metrics_from_doc) hanno la priorità.
        """
        self.log_status(f"🌐 [Tool 2.5] Arricchimento metriche VC per '{entity_name}' da web", "info")

        # 1. Cerca informazioni web sulle metriche
        query = f"VC metrics, funding, and team for {entity_name}: ARR, funding rounds, valuation, TAM, founders, team size"
        try:
            # Usiamo un search tool dedicato per 5 risultati focalizzati
            search_depth = "advanced" if is_deep_search else "basic"
            max_results = 10 if is_deep_search else 5  # 5 risultati per 'normal', 10 per 'deep'

            self.log_status(f"  [Tool 2.5] Avvio ricerca metriche (Depth: {search_depth})", "info")

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

        # 2. Consolida il contesto web
        web_context = "\n\n".join([res['content'] for res in search_results if res.get('content')])
        if answer:
            web_context += f"\n\n--- Riepilogo Web ---\n{answer}"

        if not web_context.strip():
            self.log_status("⚠️ [Tool 2.5] Nessun contenuto web trovato per le metriche", "warning")
            return metrics_from_doc  # Fallback

        # 3. Estrai metriche dal contesto web usando lo stesso schema
        try:
            extractor_chain = self.llm_pro.with_structured_output(VCMetricsProfile)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Sei un analista VC. Estrai TUTTE le metriche VC dal contesto web fornito. 
Segui lo schema JSON. Usa solo i dati forniti nel contesto.
Ignora i campi se non trovi informazioni specifiche."""),
                ("user", "Entità: {entity}\n\nContesto Web:\n\n{text}")
            ])
            pipeline = prompt | extractor_chain
            metrics_from_web = pipeline.invoke({"entity": entity_name, "text": web_context})
            self.log_status("  ✅ [Tool 2.5] Estrazione metriche da web completata", "success")
        except Exception as e:
            self.log_status(f"⚠️ [Tool 2.5] Estrazione metriche da web fallita: {e}", "warning")
            return metrics_from_doc  # Fallback

        # 4. Mergia i due profili
        final_metrics = self._merge_vc_profiles(metrics_from_doc, metrics_from_web)

        return final_metrics

    def generate_requirements_match_report(self, entity_name: str, requirements_list: str,
                                               rag_context: str) -> str:
        """
        Genera un report di Gap Analysis basato ESCLUSIVAMENTE sui documenti interni.
        Nessun dato esterno, nessun fact-checking web.
        """
        self.log_status("📋 Generazione Gap Analysis (Solo Documenti Interni)", "info")

        prompt = ChatPromptTemplate.from_messages([
                ("system", """Sei un Auditor Tecnico rigoroso. 
    Stai eseguendo una **Gap Analysis** per verificare se i documenti forniti soddisfano una lista di requisiti.

    **FONTI DATI**:
    Usa ESCLUSIVAMENTE il "Contesto Documentale" fornito qui sotto. Non usare conoscenze esterne o cercare sul web.

    **COMPITO**:
    Per ogni requisito nella lista:
    1. Cerca prove esplicite nel contesto documentale.
    2. Determina lo stato: ✅ SODDISFATTO, ⚠️ PARZIALE, o ❌ NON TROVATO.
    3. Cita la frase esatta dal documento che prova la soddisfazione.

    **OUTPUT RICHIESTO (Markdown)**:

    # 🧩 Gap Analysis Record: {entity}

    ## 📊 Sintesi
    [Breve riassunto della copertura: es. "Documentazione copre 8/10 requisiti..."]

    ## 📝 Dettaglio Verifica

    | Requisito | Stato | Evidenza nel Documento | Note/Gap |
    |-----------|-------|------------------------|----------|
    | [Nome Req] | [Stato] | [Citazione Esatta o "Nessuna menzione"] | [Dettagli] |
    ...

    ## 🏁 Conclusioni
    [Verdetto basato solo sui documenti]
    """),
                ("user", """Entità: {entity}

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
            return f"Errore: {e}"


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

def main():
    """Test CLI."""
    print("Questo script è pensato per essere importato da 'app.py'.")


if __name__ == "__main__":
    main()