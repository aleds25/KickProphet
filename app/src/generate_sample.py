import pandas as pd
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumiamo che lo script sia in app/src, quindi risaliamo a app/data/processed
INPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'processed', 'KICKSTARTER_TRAIN.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'processed', 'KICKSTARTER_TRAIN_SAMPLE_5000.csv')

def generate_sample():
    print(f"📂 Caricamento file sorgente: {os.path.basename(INPUT_FILE)}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERRORE: Il file {INPUT_FILE} non esiste.")
        print("   Assicurati di aver eseguito il notebook 2_Preprocessing.ipynb prima.")
        return

    try:
        # Carichiamo il dataset
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        total_rows = len(df)
        print(f"📊 Righe totali trovate: {total_rows}")

        # Se ci sono meno di n righe, prendiamo tutto
        sample_size = min(5000, total_rows)
        
        # Generiamo il campione casuale (random_state fissa il risultato per riproducibilità)
        print(f"🎲 Estrazione di {sample_size} righe casuali...")
        df_sample = df.sample(n=sample_size, random_state=42)

        # Salvataggio
        df_sample.to_csv(OUTPUT_FILE, index=False)
        
        print(f"✅ FILE SALVATO CORRETTAMENTE!")
        print(f"   Percorso: {os.path.abspath(OUTPUT_FILE)}")
        print(f"   Shape campione: {df_sample.shape}")

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    generate_sample()
