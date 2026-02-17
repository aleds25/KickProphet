import pandas as pd
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Input: Il file unito che abbiamo creato prima
INPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'processed', 'KICKSTARTER_RAW_ALL_MERGED.csv')
# Output: Il file pulito (ancora in processed, ma con un nome diverso)
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'processed', 'KICKSTARTER_CLEAN_BASE.csv')

print("--- 🚀 AVVIO DEDUPLICAZIONE PERSONALIZZATA ---")

# 1. Caricamento Ottimizzato
# Usiamo dtype=str per non far confondere pandas con colonne miste
print(f"📂  Lettura file: {os.path.basename(INPUT_FILE)}...")
try:
    df = pd.read_csv(INPUT_FILE, low_memory=False, dtype=str)
except FileNotFoundError:
    print(f"❌ ERRORE: Non trovo il file in {INPUT_FILE}.")
    exit()

print(f"📊  Righe totali grezze: {len(df)}")

# 2. Conversione Colonne Chiave (Personalizzazione)
# Ho visto dal tuo CSV che hai 'state_changed_at'. Usiamo quella per capire qual è la riga più recente.
print("⚙️  Conversione date e ID...")

df['id'] = pd.to_numeric(df['id'], errors='coerce') # Se l'ID è sporco, diventa NaN

# Logica personalizzata: Se c'è 'state_changed_at' usiamo quella, altrimenti 'deadline'
if 'state_changed_at' in df.columns:
    df['sort_date'] = pd.to_numeric(df['state_changed_at'], errors='coerce')
elif 'deadline' in df.columns:
    df['sort_date'] = pd.to_numeric(df['deadline'], errors='coerce')
else:
    # Caso estremo: non hai date. Usiamo l'indice (meno preciso ma funziona)
    print("⚠️  Attenzione: Nessuna colonna data trovata. Uso l'indice per ordinare.")
    df['sort_date'] = df.index

# 3. Ordinamento e Pulizia (Il cuore dello script)
print("🧹  Rimozione duplicati (Teniamo l'ultima versione del progetto)...")

# Ordiniamo per ID e Data crescente. L'ultima riga sarà la più recente.
df = df.sort_values(by=['id', 'sort_date'], ascending=[True, True])

# Rimuoviamo i duplicati basandoci SOLO sull'ID.
# keep='last' significa: tieni quella che sta in fondo (la più recente).
df_clean = df.drop_duplicates(subset='id', keep='last').copy()

print(f"📉  Righe uniche rimaste: {len(df_clean)}")

# 4. Filtro Target
# Il prof vuole un classificatore binario? Togliamo stati ambigui.
print("🎯  Filtro stati (Solo successful vs failed)...")
# Nota: 'state' potrebbe essere sporco, facciamo strip() e lower() per sicurezza se sono stringhe
if 'state' in df_clean.columns:
     df_clean = df_clean[df_clean['state'].astype(str).str.lower().isin(['successful', 'failed'])].copy()
     print(f"📉  Righe dopo filtro stati: {len(df_clean)}")
else:
    print("⚠️  Attenzione: Colonna 'state' non trovata. Salto il filtro stati.")

# Rimuoviamo la colonna temporanea 'sort_date' se l'abbiamo creata noi
if 'sort_date' in df_clean.columns and 'sort_date' not in df.columns: # check if it was original
     # In realtà l'abbiamo creata noi alla riga 30/32/35, quindi la rimuoviamo sempre
    df_clean.drop(columns=['sort_date'], inplace=True, errors='ignore')


print(f"✅  DATASET PULITO PRONTO: {len(df_clean)} progetti unici.")

# 5. Export
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"💾  Salvato in: {os.path.relpath(OUTPUT_FILE, BASE_DIR)}")