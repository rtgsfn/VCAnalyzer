from pydantic import BaseModel, Field
from typing import List, Optional

# Questo file ora contiene solo gli schemi Pydantic
# per definire la struttura dei dati estratti.

class RelazioneFondata(BaseModel):
    """Informazioni su una startup fondata da una persona."""
    persona: str = Field(description="Nome del fondatore")
    startup: str = Field(description="Nome della startup fondata")
    ruolo: Optional[str] = Field(description="Ruolo della persona, es. CEO, CTO", default="Fondatore")
    data_evento: Optional[str] = Field(description="La data (Anno o Mese-Anno) in cui è avvenuta la fondazione.")

class RelazioneInvestimento(BaseModel):
    """Informazioni su un investimento in una startup."""
    investitore: str = Field(description="Nome del fondo VC o persona che investe")
    startup: str = Field(description="Nome della startup che riceve l'investimento")
    somma_M: Optional[float] = Field(description="Somma in milioni (es. 5.5)")
    tipo_round: Optional[str] = Field(description="Tipo di round, es. Seed, Series A")
    data_evento: Optional[str] = Field(description="La data (Anno o Mese-Anno) in cui è avvenuto l'investimento.")

class RelazioneFallimento(BaseModel):
    """Informazioni su un fallimento o problema gestionale."""
    entita: str = Field(description="Nome della startup o persona")
    status: str = Field(description="Descrizione del problema, es. 'fallita', 'problemi gestionali'")
    data_evento: Optional[str] = Field(description="La data (Anno o Mese-Anno) in cui è stato riportato il problema.")

class KnowledgeGraph(BaseModel):
    """Il contenitore per tutte le relazioni estratte dal testo."""
    fondazioni: List[RelazioneFondata]
    investimenti: List[RelazioneInvestimento]
    fallimenti: List[RelazioneFallimento]

class Claim(BaseModel):
    """Rappresenta una singola affermazione fattuale trovata in un testo."""
    soggetto: str = Field(description="L'entità (persona, azienda, prodotto) a cui si riferisce l'affermazione.")
    affermazione: str = Field(description="L'affermazione fattuale specifica (es. 'ha un PhD a Berkeley', 'ha raccolto 10M$').")

class DocumentClaims(BaseModel):
    """Un contenitore per tutte le affermazioni fattuali estratte da un documento."""
    claims: List[Claim]