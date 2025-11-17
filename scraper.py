import trafilatura
import requests  # Importiamo la nuova libreria
import json

# Definiamo un "header" per mascherarci da browser Chrome su Windows
# Questo è il trucco fondamentale per superare i blocchi
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}


def scrape_article_text(url: str) -> str:
    """
    Scarica una pagina web da un URL "mascherandosi" da browser
    e ne estrae il testo principale dell'articolo.
    """
    print(f"\n--- Avvio Scraper sull'URL: {url} ---")

    # --- BLOCCO MODIFICATO ---
    downloaded_html = None
    try:
        # 1. Scarica la pagina usando 'requests' con i nostri header
        response = requests.get(url, headers=HEADERS, timeout=10)

        # Controlla se la richiesta è andata a buon fine (codice 200)
        if response.status_code == 200:
            downloaded_html = response.text  # Prendiamo l'HTML
            print("Pagina scaricata con successo.")
        else:
            print(f"Errore: Impossibile scaricare la pagina. Codice di stato: {response.status_code}")
            return None

    except requests.RequestException as e:
        print(f"Errore: Richiesta fallita. {e}")
        return None
    # --- FINE BLOCCO MODIFICATO ---

    if not downloaded_html:
        print("Errore: Download fallito.")
        return None

    print("Estrazione del testo in corso...")

    # 2. Estrai il testo principale (questa parte non cambia)
    # Ora passiamo l'HTML scaricato a 'trafilatura.extract'
    extracted_text = trafilatura.extract(
        downloaded_html,
        include_links=False,
        output_format='txt'
    )

    if not extracted_text:
        print("Errore: Estrazione fallita. La pagina potrebbe essere vuota o protetta.")
        return None

    print("--- Estrazione Testo Completata ---")
    return extracted_text


# --- Sezione di Test (non cambia) ---
if __name__ == "__main__":
    """
    Testiamo lo scraper su un vero articolo di TechCrunch.
    """

    TEST_URL = "https://techcrunch.com/2025/11/09/apple-reportedly-plans-ambitious-satellite-powered-iphone-features/"

    testo_estratto = scrape_article_text(TEST_URL)

    if testo_estratto:
        print("\n--- RISULTATO DELLO SCRAPING (primi 500 caratteri) ---")
        print(testo_estratto[:500] + "...")
    else:
        print("\n--- Test fallito. ---")