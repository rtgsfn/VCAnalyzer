"""
vc_metrics.py - Framework per Estrazione e Validazione Metriche VC

Questo modulo definisce le metriche chiave che i VC analizzano durante la due diligence.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

# ============================================================================
# ENUMERAZIONI PER CLASSIFICAZIONE
# ============================================================================

class BusinessModel(str, Enum):
    """Modelli di business comuni nel VC."""
    SAAS = "SaaS"
    MARKETPLACE = "Marketplace"
    ECOMMERCE = "E-commerce"
    HARDWARE = "Hardware"
    FINTECH = "Fintech"
    CONSUMER_APP = "Consumer App"
    ENTERPRISE = "Enterprise Software"
    DEEPTECH = "Deep Tech"
    BIOTECH = "Biotech"
    OTHER = "Other"


class FundingStage(str, Enum):
    """Stage di finanziamento."""
    PRE_SEED = "Pre-Seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    SERIES_D_PLUS = "Series D+"
    UNKNOWN = "Unknown"


class MetricStatus(str, Enum):
    """Status di verifica di una metrica."""
    VERIFIED = "Verificata"
    UNVERIFIED = "Non Verificata"
    CONFLICTING = "Conflittuale"
    MISSING = "Mancante"


# ============================================================================
# METRICHE FINANZIARIE (SaaS-focused)
# ============================================================================

class SaaSMetrics(BaseModel):
    """Metriche critiche per startup SaaS."""

    # Revenue Metrics
    arr: Optional[float] = Field(None, description="Annual Recurring Revenue in $M")
    mrr: Optional[float] = Field(None, description="Monthly Recurring Revenue in $K")
    revenue_growth_rate: Optional[float] = Field(None, description="YoY Revenue Growth Rate (%)")

    # Efficiency Metrics
    gross_margin: Optional[float] = Field(None, description="Gross Margin (%)")
    ltv_cac_ratio: Optional[float] = Field(None, description="Lifetime Value / Customer Acquisition Cost")
    cac_payback_months: Optional[int] = Field(None, description="CAC Payback Period (months)")

    # Growth Metrics
    net_retention_rate: Optional[float] = Field(None, description="Net Dollar Retention (%)")
    gross_retention_rate: Optional[float] = Field(None, description="Gross Retention (%)")
    magic_number: Optional[float] = Field(None, description="Sales Efficiency (New ARR / S&M Spend)")

    # Burn & Runway
    monthly_burn: Optional[float] = Field(None, description="Monthly Burn Rate in $K")
    runway_months: Optional[int] = Field(None, description="Runway (months)")

    # Rule of 40 (Growth Rate + Profit Margin)
    rule_of_40: Optional[float] = Field(None, description="Growth Rate + EBITDA Margin")

    # Status per ogni metrica
    metrics_status: Optional[Dict[str, MetricStatus]] = Field(default_factory=dict)


# ============================================================================
# METRICHE DI TRACTION
# ============================================================================

class TractionMetrics(BaseModel):
    """Metriche di traction e crescita."""

    # User Metrics
    total_users: Optional[int] = Field(None, description="Totale utenti/clienti")
    active_users_mau: Optional[int] = Field(None, description="Monthly Active Users")
    paying_customers: Optional[int] = Field(None, description="Clienti paganti")

    # Growth Metrics
    user_growth_rate: Optional[float] = Field(None, description="User Growth Rate MoM (%)")
    revenue_per_customer: Optional[float] = Field(None, description="ARPU - Average Revenue Per User ($)")

    # Engagement
    dau_mau_ratio: Optional[float] = Field(None, description="DAU/MAU Ratio (stickiness)")
    nps_score: Optional[int] = Field(None, description="Net Promoter Score")

    # Enterprise Metrics
    enterprise_customers: Optional[int] = Field(None, description="Numero clienti Enterprise (>100K ARR)")
    average_contract_value: Optional[float] = Field(None, description="ACV - Average Contract Value ($K)")

    metrics_status: Dict[str, MetricStatus] = Field(default_factory=dict)


# ============================================================================
# METRICHE DI MERCATO
# ============================================================================

class MarketMetrics(BaseModel):
    """Metriche di mercato e posizionamento."""

    tam: Optional[float] = Field(None, description="Total Addressable Market ($B)")
    sam: Optional[float] = Field(None, description="Serviceable Addressable Market ($B)")
    som: Optional[float] = Field(None, description="Serviceable Obtainable Market ($M)")

    market_share: Optional[float] = Field(None, description="Current Market Share (%)")
    market_growth_rate: Optional[float] = Field(None, description="Market CAGR (%)")

    competitive_position: Optional[str] = Field(None, description="Posizione competitiva (Leader/Challenger/Niche)")

    metrics_status: Dict[str, MetricStatus] = Field(default_factory=dict)


# ============================================================================
# METRICHE DI TEAM
# ============================================================================

class TeamMember(BaseModel):
    """Informazioni su un membro del team."""
    name: str
    role: str
    background: Optional[str] = Field(None, description="Ex. 'Ex-Google, Stanford PhD'")
    previous_exits: Optional[int] = Field(None, description="Numero di exit precedenti")
    years_experience: Optional[int] = Field(None, description="Anni di esperienza nel settore")
    verified: bool = Field(False, description="Background verificato pubblicamente")


class TeamMetrics(BaseModel):
    """Metriche del team di fondatori."""

    founders: Optional[List[TeamMember]] = Field(default_factory=list)
    total_team_size: Optional[int] = Field(None, description="Dimensione totale del team")
    engineering_team_size: Optional[int] = Field(None, description="Dimensione team engineering")

    # Founder Market Fit
    years_in_industry: Optional[int] = Field(None, description="Anni dei founder nel settore")
    previous_startups: Optional[int] = Field(None, description="Startup precedenti dei founder")
    technical_founders: Optional[int] = Field(None, description="Numero founder tecnici")

    # Advisor/Board Quality
    notable_advisors: Optional[List[str]] = Field(default_factory=list)
    notable_investors: Optional[List[str]] = Field(default_factory=list)


# ============================================================================
# METRICHE DI FUNDRAISING
# ============================================================================

class FundraisingRound(BaseModel):
    """Dettagli di un round di fundraising."""
    stage: FundingStage
    amount: Optional[float] = Field(None, description="Importo raccolto ($M)")
    valuation: Optional[float] = Field(None, description="Post-money valuation ($M)")
    date: Optional[str] = Field(None, description="Data del round")
    lead_investor: Optional[str] = Field(None, description="Lead investor")
    other_investors: Optional[List[str]] = Field(default_factory=list)  # <-- Aggiungi Optional
    verified: Optional[bool] = Field(False)


class FundraisingMetrics(BaseModel):
    """Storia di fundraising."""

    rounds: Optional[List[FundraisingRound]] = Field(default_factory=list) # <-- Aggiungi Optional
    total_raised: Optional[float] = Field(None, description="Totale raccolto ($M)")
    last_valuation: Optional[float] = Field(None, description="Ultima valuation ($M)")

    # Investor Quality Score (calcolato)
    tier1_investors: Optional[int] = Field(0, description="Numero investitori Tier 1 (Sequoia, a16z, etc.)") # <-- Aggiungi Optional


# ============================================================================
# CONTAINER COMPLETO
# ============================================================================

class VCMetricsProfile(BaseModel):
    """Profilo completo di metriche VC per un'entità."""

    entity_name: str
    business_model: Optional[BusinessModel] = None
    current_stage: Optional[FundingStage] = None

    # Metriche per categoria
    saas_metrics: Optional[SaaSMetrics] = None
    traction_metrics: Optional[TractionMetrics] = None
    market_metrics: Optional[MarketMetrics] = None
    team_metrics: Optional[TeamMetrics] = None
    fundraising_metrics: Optional[FundraisingMetrics] = None

    # Metadata
    last_updated: Optional[str] = None
    data_sources: Optional[List[str]] = Field(default_factory=list)


# ============================================================================
# HELPER FUNCTIONS PER CALCOLI DERIVATI
# ============================================================================

def calculate_rule_of_40(revenue_growth_rate: float, ebitda_margin: float) -> float:
    """
    Calcola la Rule of 40.
    Rule of 40: Revenue Growth Rate + Profit Margin ≥ 40%

    Score > 40 = Eccellente
    Score 20-40 = Buono
    Score < 20 = Preoccupante
    """
    return revenue_growth_rate + ebitda_margin


def calculate_magic_number(new_arr: float, sales_marketing_spend: float) -> float:
    """
    Calcola il Magic Number (Sales Efficiency).
    Magic Number = New ARR / S&M Spend

    > 1.0 = Eccellente efficienza di vendita
    0.75-1.0 = Buona efficienza
    < 0.75 = Inefficiente
    """
    if sales_marketing_spend == 0:
        return 0
    return new_arr / sales_marketing_spend


def calculate_ltv_cac_ratio(ltv: float, cac: float) -> float:
    """
    Calcola LTV/CAC ratio.

    > 3.0 = Eccellente
    2.0-3.0 = Buono
    < 2.0 = Preoccupante
    """
    if cac == 0:
        return 0
    return ltv / cac


def assess_t2d3_trajectory(arr_history: List[float]) -> bool:
    """
    Verifica se la startup è su una traiettoria T2D3.
    T2D3 = Triple, Triple, Double, Double, Double

    Path to $100M ARR in 5-7 anni.
    """
    if len(arr_history) < 2:
        return False

    # Verifica se la crescita è almeno 2x year-over-year
    for i in range(1, len(arr_history)):
        growth = arr_history[i] / arr_history[i - 1] if arr_history[i - 1] > 0 else 0
        if growth < 2.0:
            return False
    return True


# ============================================================================
# BENCHMARK TIERS (per confronto)
# ============================================================================

SAAS_BENCHMARKS = {
    "arr_growth_rate": {
        "seed": {"excellent": 300, "good": 200, "acceptable": 100},
        "series_a": {"excellent": 200, "good": 150, "acceptable": 100},
        "series_b": {"excellent": 150, "good": 100, "acceptable": 75},
    },
    "net_retention_rate": {
        "excellent": 120,
        "good": 110,
        "acceptable": 100,
        "poor": 90
    },
    "ltv_cac_ratio": {
        "excellent": 4.0,
        "good": 3.0,
        "acceptable": 2.0,
        "poor": 1.5
    },
    "cac_payback_months": {
        "excellent": 6,
        "good": 12,
        "acceptable": 18,
        "poor": 24
    },
    "gross_margin": {
        "excellent": 80,
        "good": 70,
        "acceptable": 60,
        "poor": 50
    }
}

TIER_1_VCS = [
    "Sequoia Capital", "Andreessen Horowitz", "Accel", "Benchmark",
    "Greylock Partners", "Lightspeed Venture Partners", "Index Ventures",
    "Founders Fund", "General Catalyst", "NEA", "Kleiner Perkins"
]


def get_benchmark_assessment(metric_name: str, value: float, stage: str = "series_a") -> str:
    """
    Restituisce un assessment qualitativo di una metrica vs benchmark.

    Returns: "Excellent" | "Good" | "Acceptable" | "Poor" | "Unknown"
    """
    benchmarks = SAAS_BENCHMARKS.get(metric_name)

    if not benchmarks:
        return "Unknown"

    # Se ha stage-specific benchmarks
    if isinstance(benchmarks, dict) and stage in benchmarks:
        stage_bench = benchmarks[stage]
        if value >= stage_bench["excellent"]:
            return "Excellent"
        elif value >= stage_bench["good"]:
            return "Good"
        elif value >= stage_bench["acceptable"]:
            return "Acceptable"
        else:
            return "Poor"

    # Altrimenti usa benchmarks generali
    if "excellent" in benchmarks:
        # Per metriche dove higher is better
        if value >= benchmarks["excellent"]:
            return "Excellent"
        elif value >= benchmarks["good"]:
            return "Good"
        elif value >= benchmarks.get("acceptable", 0):
            return "Acceptable"
        else:
            return "Poor"

    return "Unknown"


# Aggiungi questi codici in fondo a vc_metrics.py

# ============================================================================
# SCHEMI SPECIFICI PER SETTORE
# ============================================================================

# Pharma & Biotech Metrics
class PharmaRDMentrics(BaseModel):
    clinical_trial_phase: Optional[str] = Field(None, description="Fase di trial clinico attuale (Preclinical, I, II, III, NDA/BLA).")
    fda_approval_status: Optional[str] = Field(None, description="Stato di approvazione regolatoria (Approved, Pending, Denied).")
    time_to_market_years: Optional[float] = Field(None, description="Tempo stimato o trascorso per l'approvazione (anni).")
    patent_expiry_date: Optional[str] = Field(None, description="Data di scadenza del brevetto chiave.")
    efficacy_data: Optional[str] = Field(None, description="Dati chiave di efficacia (es. ORR %, PFS months).")
    rd_burn_rate_m: Optional[float] = Field(None, description="Burn rate mensile per R&D in $M.")

class PharmaMetricsProfile(BaseModel):
    """Profilo di metriche per Pharma/Biotech."""
    entity_name: str
    rd_metrics: Optional[PharmaRDMentrics] = None
    team_metrics: Optional[TeamMetrics] = None # Riutilizzo del Team (chi ha fondato, background)
    fundraising_metrics: Optional[FundraisingMetrics] = None # Riutilizzo del Fundraising

# Real Estate Metrics
class REFinancialMetrics(BaseModel):
    cap_rate: Optional[float] = Field(None, description="Capitalization Rate (Tasso di rendimento annuale) (%).")
    cash_on_cash: Optional[float] = Field(None, description="Cash-on-Cash Return (%).")
    irr: Optional[float] = Field(None, description="Internal Rate of Return (IRR) (%).")
    occupancy_rate: Optional[float] = Field(None, description="Tasso di occupazione attuale (%).")
    net_operating_income: Optional[float] = Field(None, description="Reddito Operativo Netto (NOI) in $M.")

class REMetricsProfile(BaseModel):
    """Profilo di metriche per Real Estate."""
    entity_name: str
    re_metrics: Optional[REFinancialMetrics] = None
    market_metrics: Optional[MarketMetrics] = None # Riutilizzo del Market (TAM, SOM)
    team_metrics: Optional[TeamMetrics] = None # Riutilizzo

# Legal / M&A Metrics
class LegalRiskMetrics(BaseModel):
    gdpr_compliance: Optional[bool] = Field(None, description="Verifica esplicita di conformità GDPR/CCPA.")
    iso_soc_certified: Optional[str] = Field(None, description="Certificazioni di sicurezza/compliance (es. ISO 27001, SOC 2).")
    pending_litigation_count: Optional[int] = Field(None, description="Numero di contenziosi legali pendenti.")
    ip_status: Optional[str] = Field(None, description="Stato della Proprietà Intellettuale (Brevettato, In attesa, Nessuno).")
    change_of_control_clause: Optional[bool] = Field(None, description="Presenza di clausole di Change of Control in contratti chiave.")

class LegalMetricsProfile(BaseModel):
    """Profilo di metriche per Legal/M&A Due Diligence."""
    entity_name: str
    legal_metrics: Optional[LegalRiskMetrics] = None
    team_metrics: Optional[TeamMetrics] = None # Riutilizzo

# Unione per Type Hinting dinamico
SectorMetricsProfile = VCMetricsProfile | REMetricsProfile | PharmaMetricsProfile | LegalMetricsProfile