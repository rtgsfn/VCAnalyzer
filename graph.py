import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from typing import Dict, Any

# Importiamo gli schemi Pydantic per la scrittura
try:
    from extractor import KnowledgeGraph, RelazioneFondata, RelazioneInvestimento, RelazioneFallimento
except ImportError:
    print("Avviso: Impossibile importare i modelli da extractor.py per il test di graph.py.")


    class KnowledgeGraph:
        pass


    class RelazioneFondata:
        pass


    class RelazioneInvestimento:
        pass


    class RelazioneFallimento:
        pass


class GraphTool:
    """
    Strumento potenziato per leggere E SCRIVERE sul Knowledge Graph Neo4j.
    VERSIONE CORRETTA.
    """

    def __init__(self, uri, user, password):
        """Inizializza il driver Neo4j."""
        print("Connessione a Neo4j...")
        try:
            self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Connessione a Neo4j riuscita.")
        except Exception as e:
            print(f"Errore di connessione a Neo4j: {e}")
            self.driver = None

    def close(self):
        """Chiude la connessione del driver."""
        if self.driver:
            self.driver.close()
            print("Connessione a Neo4j chiusa.")

    def _execute_query(self, query, params={}):
        """Helper per eseguire query (lettura e scrittura)."""
        if not self.driver:
            return "Errore: Driver Neo4j non inizializzato."
        try:
            records, _, _ = self.driver.execute_query(query, params, database_="neo4j")
            return records
        except Exception as e:
            print(f"Errore durante l'esecuzione della query: {e}")
            return f"Errore: {e}"

    # --- FUNZIONI DI SCRITTURA (CICLO 1) ---

    def import_extracted_data(self, data: KnowledgeGraph):
        """Scrive i dati estratti (con date) nel grafo."""
        print("\n--- AVVIO SCRITTURA DATI ESTRATTI SU NEO4J ---")

        # 1. Scrivi Fondazioni
        for rel in data.fondazioni:
            query = """
            MERGE (p:Persona {name: $persona_name})
            MERGE (s:Startup {name: $startup_name})
            MERGE (p)-[r:HA_FONDATO]->(s)
            ON CREATE SET r.ruolo = $ruolo, r.data_evento = $data_evento
            """
            self._execute_query(query, {
                "persona_name": rel.persona, "startup_name": rel.startup,
                "ruolo": rel.ruolo, "data_evento": rel.data_evento
            })

        # 2. Scrivi Investimenti
        for rel in data.investimenti:
            query = """
            MERGE (i:Entita {name: $investitore_name}) ON CREATE SET i:Investitore
            MERGE (s:Startup {name: $startup_name})
            MERGE (i)-[r:HA_INVESTITO_IN]->(s)
            ON CREATE SET r.somma_M = $somma, r.tipo_round = $tipo_round, r.data_evento = $data_evento
            """
            self._execute_query(query, {
                "investitore_name": rel.investitore, "startup_name": rel.startup,
                "somma": rel.somma_M, "tipo_round": rel.tipo_round, "data_evento": rel.data_evento
            })

        # 3. Scrivi Fallimenti/Problemi
        for rel in data.fallimenti:
            query = """
            MERGE (e:Entita {name: $entita_name}) ON CREATE SET e:Startup
            SET e.status = $status, e.status_data = $data_evento
            """
            self._execute_query(query, {
                "entita_name": rel.entita, "status": rel.status, "data_evento": rel.data_evento
            })

        print("--- SCRITTURA COMPLETATA ---")

    # --- FUNZIONI DI LETTURA (CICLO 2 / Graph RAG) ---
    def _format_path_as_text(self, path) -> str:
        """Helper per formattare un 'path' di Neo4j in testo, INCLUSE le proprietà."""

        nodes = path.nodes  # Lista di oggetti Nodo
        rels = path.relationships  # Lista di oggetti Relazione

        formatted_path = ""
        formatted_path += f"({nodes[0]['name']})"

        for i, rel in enumerate(rels):
            # Gestione compatibile con diverse versioni del driver Neo4j
            try:
                # Metodo 1: accesso diretto a properties (versioni recenti)
                if hasattr(rel, 'properties'):
                    props = dict(rel.properties)
                # Metodo 2: l'oggetto è già un dizionario-like
                elif hasattr(rel, 'items'):
                    props = dict(rel)
                # Metodo 3: accesso tramite chiavi
                else:
                    props = {key: rel[key] for key in rel.keys() if key != 'type'}
            except Exception:
                props = {}

            # Pulisci le proprietà nulle per leggibilità
            props_str = ", ".join([f"{k}: '{v}'" for k, v in props.items() if v is not None])

            # Ottieni il tipo della relazione in modo robusto
            rel_type = rel.type if hasattr(rel, 'type') else str(rel.__class__.__name__)
            end_node = nodes[i + 1]['name']

            # Aggiungi la relazione con le sue proprietà
            if props_str:
                formatted_path += f" -[:{rel_type} {{{props_str}}}]-> ({end_node})"
            else:
                formatted_path += f" -[:{rel_type}]-> ({end_node})"

        return formatted_path

    def get_semantic_context(self, entity_name: str, max_hops: int = 2, limit: int = 10) -> str:
        """Recupera il contesto "Graph RAG" per un'entità."""
        print(f"\n--- Recupero Contesto Graph RAG per: '{entity_name}' (max {max_hops} hop) ---")

        query = f"""
        MATCH path = (n {{name: $name}})-[*1..{max_hops}]-(m)
        RETURN path
        LIMIT {limit}
        """
        params = {"name": entity_name}
        records = self._execute_query(query, params)

        if not records or isinstance(records, str):
            print("Nessun contesto trovato nel grafo.")
            return "Nessun contesto trovato nel Knowledge Graph."

        facts = []
        for record in records:
            path = record["path"]
            # Chiama la funzione helper CORRETTA
            facts.append(self._format_path_as_text(path))

        unique_facts = list(set(facts))
        context_text = "\n".join(unique_facts)
        print(f"Contesto testuale (temporale) dal grafo:\n{context_text}")
        return context_text

    # ---
    # --- FIX 2: Funzione get_graph_visualization_data CORRETTA ---
    # ---
    def get_graph_visualization_data(self, entity_name: str, max_hops: int = 2):
        """Recupera i nodi e gli archi per la visualizzazione della GUI."""
        print(f"\n--- Recupero Dati di Visualizzazione Grafo per: '{entity_name}' ---")

        query = f"""
        MATCH path = (n {{name: $name}})-[*1..{max_hops}]-(m)
        RETURN nodes(path) as nodes, relationships(path) as rels
        LIMIT 20
        """
        params = {"name": entity_name}
        records = self._execute_query(query, params)

        if not records or isinstance(records, str):
            return [], []

        nodes_set = set()
        edges_set = set()

        for record in records:
            nodes_data = record["nodes"]
            rels_data = record["rels"]

            for node in nodes_data:
                node_name = node['name']
                labels = [label for label in node.labels if label != 'Entita']
                label_str = ":".join(labels) if labels else "Entita"
                is_target = (node_name == entity_name)
                nodes_set.add((node_name, label_str, "red" if is_target else "#666666"))

            for rel in rels_data:
                # Gestione robusta delle proprietà della relazione
                try:
                    if hasattr(rel, 'properties'):
                        props = dict(rel.properties)
                    elif hasattr(rel, 'items'):
                        props = dict(rel)
                    else:
                        props = {key: rel[key] for key in rel.keys() if key != 'type'}
                except Exception:
                    props = {}

                # Ottieni nodi start/end in modo robusto
                if hasattr(rel, 'start_node') and hasattr(rel, 'end_node'):
                    start_node = rel.start_node['name']
                    end_node = rel.end_node['name']
                else:
                    # Fallback: usa la posizione nell'array
                    start_node = nodes_data[0]['name']
                    end_node = nodes_data[-1]['name']

                rel_type = rel.type if hasattr(rel, 'type') else str(rel.__class__.__name__)

                label = rel_type
                if 'data_evento' in props and props['data_evento']:
                    label += f" ({props['data_evento']})"
                if 'somma_M' in props and props['somma_M']:
                    label += f" ({props['somma_M']}M)"

                edges_set.add((start_node, end_node, label))

        nodes = [{"id": n[0], "label": f"{n[1]}\n{n[0]}", "color": n[2], "size": 15} for n in nodes_set]
        edges = [{"source": e[0], "target": e[1], "label": e[2]} for e in edges_set]

        return nodes, edges


# --- Sezione di Test (per eseguire 'python graph.py' da solo) ---
def main():
    print("--- Test CLI per graph.py ---")
    load_dotenv()
    graph_tool = GraphTool(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    if not graph_tool.driver:
        return

    print("Pulizia del DB...")
    graph_tool._execute_query("MATCH (n) DETACH DELETE n")
    print("Popolamento con dati di test...")
    fake_data = KnowledgeGraph(
        fondazioni=[RelazioneFondata(persona='Mario Rossi', startup='TestStartup', ruolo='CEO', data_evento='2023')],
        investimenti=[RelazioneInvestimento(investitore='TestVC', startup='TestStartup', somma_M=5.0, tipo_round='Seed',
                                            data_evento='2024')],
        fallimenti=[]
    )
    graph_tool.import_extracted_data(fake_data)

    print("\n--- Test Lettura (Graph RAG) ---")
    context = graph_tool.get_semantic_context("TestStartup")
    print(f"Risultato Test RAG:\n{context}")

    print("\n--- Test Lettura (Visualizzazione) ---")
    nodes, edges = graph_tool.get_graph_visualization_data("TestStartup")
    print(f"Nodi per GUI: {nodes}")
    print(f"Archi per GUI: {edges}")

    graph_tool.close()


if __name__ == "__main__":
    main()