import pandas as pd
import glob
import os

# Configurazione path relativi allo script
# Lo script si trova in app/src, quindi torniamo indietro di uno step per andare in app/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

# Assicuriamoci che la cartella processed esista
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_FILENAME = "KICKSTARTER_RAW_ALL_MERGED.csv"
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME)

print("--- 🚀 INIZIO MERGE GREZZO (NESSUNA PULIZIA) ---")
print(f"📂  Cerco file CSV in: {RAW_DATA_DIR}")

# 1. Trova tutti i file CSV nella cartella raw
search_pattern = os.path.join(RAW_DATA_DIR, "*.csv")
files = glob.glob(search_pattern)
print(f"📂  Trovati {len(files)} file CSV da unire.")

lista_dataframe = []

for filename in files:
    # Controllo di sicurezza, anche se l'output va in processed
    if os.path.basename(filename) == OUTPUT_FILENAME:
        continue

    try:
        print(f"   Reading {os.path.basename(filename)}...", end="\r")

        # TRUCCO FONDAMENTALE: dtype=str
        # Leggiamo tutto come testo puro.
        # Questo evita errori di "Mixed Types" e mantiene il dato fedele al 100% all'originale.
        df_temp = pd.read_csv(filename, on_bad_lines='skip', low_memory=False, dtype=str)

        lista_dataframe.append(df_temp)
    except Exception as e:
        print(f"\n   ⚠️  Errore critico su {filename}: {e}")

if not lista_dataframe:
    print("\n❌ Nessun file valido trovato in data/raw.")
    exit()

print(f"\n🔄  Sto unendo {len(lista_dataframe)} dataframe... (Richiede memoria)")
# Concatenazione pura
df_full = pd.concat(lista_dataframe, ignore_index=True)

print(f"✅  Merge completato.")
print(f"📊  Totale righe: {len(df_full)}")
print(f"    Totale colonne: {len(df_full.columns)}")

# Salvataggio
print(f"💾  Salvataggio in corso su {OUTPUT_PATH}...")
df_full.to_csv(OUTPUT_PATH, index=False)
print(f"🎉  Fatto! Hai il tuo dataset grezzo unico in: {os.path.relpath(OUTPUT_PATH, BASE_DIR)}")