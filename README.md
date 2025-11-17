Here is a comprehensive README.md file generated based on the complete project structure you provided.

-----

# 🎯 Agentic VC KRAG Analyzer

**Knowledge-Retrieval Augmented Generation (KRAG) for Investigative Due Diligence in Venture Capital**

This project is an advanced, agentic system designed to assist Venture Capital (VC) analysts in performing rapid and deep due diligence on startups.

It goes beyond simple RAG by implementing a **KRAG** (Knowledge-Graph Retrieval Augmented Generation) architecture. It synthesizes information from three distinct sources to build a comprehensive investment memo:

1.  **Private Documents (RAG):** Information from uploaded pitch decks, memos, and text (via ChromaDB).
2.  **Public Knowledge (Graph):** Structured, persistent data on companies, founders, and investments (via Neo4j).
3.  **Live Web (Augmentation):** Real-time web search for fact-checking, metric augmentation, and news (via Tavily and DuckDuckGo).

The agent extracts structured VC metrics, fact-checks claims from the pitch deck against public data, analyzes the founding team's history, and generates a full investment report including risk, feasibility, and a final recommendation.

## ✨ Core Features

  * **Agentic Workflow:** An autonomous agent (`analyzer.py`) orchestrates a multi-step analysis: metric extraction, claim extraction, fact-checking, graph population, and multi-source synthesis.
  * **Hybrid KRAG Architecture:**
      * **Vector RAG:** Uses **ChromaDB** to perform semantic search on private, user-uploaded documents (`ingest.py`).
      * **Graph RAG:** Uses **Neo4j** to store and query a persistent Knowledge Graph of public facts (founders, funding, past failures) (`graph.py`).
  * **Dynamic Knowledge Graph:** The agent actively enriches the Neo4j graph. When it analyzes a new company, it performs web searches and *writes* new, verified findings back to the graph (`run_population_cycle` in `analyzer.py`).
  * **Automated Fact-Checking:** Extracts factual claims from documents (e.g., "Our CEO has a PhD from Stanford") and automatically verifies them against the web, labeling them as `VERIFICATA`, `FALSA`, or `NON VERIFICABILE`.
  * **Deep VC Metrics Extraction:** Uses a rich set of Pydantic models (`vc_metrics.py`) to extract and analyze key performance indicators (KPIs) across SaaS, Traction, Market, Team, and Fundraising.
  * **Multi-LLM Support:** Easily configurable to use various LLM providers, including **Google (Gemini)**, **Groq (Llama 3.1)**, **Perplexity**, and local **Ollama** models.
  * **Interactive Streamlit UI:** A clean, web-based interface (`app.py`) to upload documents, run analyses, and progressively stream the results, including a final report and an interactive graph visualization.

## 🏛️ System Architecture

The system is orchestrated by the `AgenticKRAG` class, which follows a defined workflow to gather and synthesize information for the final report.

```mermaid
graph TD
    A[User] -->|1. Upload Docs & Entity Name| B(Streamlit UI - app.py)
    B -->|2. Start Analysis| C(AgenticKRAG - analyzer.py)

    subgraph "Private Data (Per-Run)"
        C -->|3. Process Docs| D(ChromaDB)
        D -->|8. RAG Context| C
    end

    subgraph "Public Data (Persistent)"
        C -->|4. Populate/Update Graph| E(Tavily Web Search)
        E -->|5. Extract Facts (LLM)| F
        F -->|6. Write Facts| G(Neo4j Graph)
        G -->|9. Graph Context| C
    end

    subgraph "Live Data (Per-Run)"
        C -->|7. Fact-Check Claims| H(Tavily / DuckDuckGo)
        H -->|10. Fact-Check Results| C
    end

    subgraph "Synthesis"
        C -->|11. Generate Report| I(LLM - Gemini/Groq/Llama)
        I -->|12. Stream Results| B
    end
```

## 🛠️ Tech Stack

  * **Frontend:** Streamlit, Streamlit-Agraph
  * **Orchestration:** LangChain, LangGraph (for the `main.py` prototype)
  * **Core Logic:** Python 3.13
  * **LLM Providers:** `langchain-google-genai`, `langchain-groq`, `langchain-openai` (for Perplexity), `langchain-community` (Ollama)
  * **Vector Database (Private RAG):** ChromaDB, HuggingFace Embeddings (`all-MiniLM-L6-v2`)
  * **Graph Database (Public KG):** Neo4j
  * **Web Search:** Tavily, DuckDuckGo-Search
  * **Web Scraping:** Trafilatura, Requests
  * **Data Modeling:** Pydantic

## 🚀 Getting Started

### 1\. Prerequisites

  * **Python 3.13** (as specified in `.idea/misc.xml`)
  * **Neo4j Database:** This project requires a running Neo4j instance. The easiest way is via [Neo4j Desktop](https://neo4j.com/download/) or Docker:
    ```bash
    docker run \
        --name vcanalyzer-neo4j \
        -p 7474:7474 -p 7687:7687 \
        -d \
        -e NEO4J_AUTH=neo4j/your-strong-password \
        neo4j:latest
    ```
  * **API Keys:** You will need API keys from Google, Groq, Perplexity, and Tavily.

### 2\. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/VCAnalyzer-master.git
    cd VCAnalyzer-master
    ```

2.  **Create a virtual environment:**

    ```bash
    python3.13 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *(Note: A `requirements.txt` was not provided, but you can create one from the imports or install the key libraries manually.)*

    ```bash
    pip install streamlit streamlit-agraph langchain langgraph \
                langchain-google-genai langchain-groq langchain-openai \
                langchain-community langchain-huggingface \
                neo4j chromadb tavily-python duckduckgo-search \
                trafilatura requests pydantic \
                langchain-chroma langchain-text-splitters
    ```

### 3\. Configuration

Create a `.env` file in the root of the project and add your credentials.

```.env
# LLM Keys
GOOGLE_API_KEY="AIza..."
GROQ_API_KEY="gsk_..."
PERPLEXITY_API_KEY="pplx-..."

# Knowledge Graph (Neo4j)
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-strong-password"

# Agent Tools
TAVILY_API_KEY="tvly-..."
```

### 4\. How to Run

1.  **Ingest Your Private Documents:**

      * Place your pitch decks, memos, or other `.txt` files into the `input_docs/` directory.
      * Run the ingestion script to populate the local ChromaDB:
        ```bash
        python ingest.py
        ```
      * This will create a `chroma_db/` folder containing the vector embeddings for your private documents.

2.  **Launch the Streamlit Application:**

      * Run the `app.py` script:
        ```bash
        streamlit run app.py
        ```

3.  **Use the Analyzer:**

      * Open the URL (e.g., `http://localhost:8501`) in your browser.
      * In the sidebar, select the LLM Provider and Model you configured.
      * In the main panel, upload the documents you want to analyze (or paste text).
      * Enter the name of the startup or entity to investigate (e.g., "AstraBio", "Figure AI").
      * Click the "🚀 Avvia Analisi Investigativa" button.
      * The agent will now run the full analysis, streaming the results (Metrics, Fact-Checking, Risk, Feasibility, Summary, and Graph) to the UI as they are generated.

## 📂 Project Structure

Here is a brief overview of the key files in the project:

```
VCAnalyzer-master/
├── app.py                  # The main Streamlit frontend application
├── analyzer.py             # The core AgenticKRAG class, orchestrating the full analysis
├── vc_metrics.py           # Pydantic models for all VC KPIs (SaaS, Traction, etc.)
├── graph.py                # The GraphTool for all Neo4j interactions (read/write)
├── extractor.py            # Pydantic models for graph relationships (KnowledgeGraph) and document Claims
├── tools.py                # Initializes the Chroma RAG retriever and DuckDuckGo tool
├── ingest.py               # Script to process /input_docs into the ChromaDB vector store
├── scraper.py              # Utility to scrape clean text from web URLs
├── config.py               # Config for graph colors, UI messages, and analysis parameters
├── main.py                 # A LangGraph-based prototype/CLI version of the agent logic
├── pipeline.py             # A separate small pipeline to populate Neo4j from a single URL
│
├── input_docs/             # --- Add your private pitch decks here ---
│   ├── pitch_FigureAI.txt
│   └── ...
│
├── chroma_db/              # (Generated) Local Chroma vector store
├── logs/                   # (Generated) Log files for each analysis run
│
└── .env                    # (You must create this) API keys and credentials
```

-----
