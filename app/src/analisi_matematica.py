import pandas as pd
import numpy as np
import os


def audit_maniacale_v4_definitiva(file_path):
    if not os.path.exists(file_path):
        print(f"❌ ERRORE: Il file '{file_path}' non è stato trovato.")
        return

    print(f"🕵️  AVVIO AUDIT DEFINITIVO: {file_path}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    # 1. VERIFICA TRIGONOMETRICA (Range [-1, 1])
    print("\n📐 VERIFICA SIN/COS (L'incubo di Excel)")
    trig_cols = [c for c in df.columns if '_sin' in c or '_cos' in c]
    if trig_cols:
        for col in trig_cols:
            c_min, c_max = df[col].min(), df[col].max()
            # Test di tolleranza per floating point
            status = "✅ OK" if -1.0001 <= c_min <= 1.0001 and -1.0001 <= c_max <= 1.0001 else "❌ ERRORE"
            print(f"   {status} | {col.ljust(20)}: [{c_min:7.4f}, {c_max:7.4f}]")
    else:
        print("   ⚠️  Attenzione: Nessuna colonna sin/cos trovata!")

    # 2. ANALISI LOG-TRANSFORMS (La "cura" per gli outlier)
    print("\n📊 ANALISI DISTRIBUZIONI LOGARITMICHE (Best Practice)")
    log_cols = ['goal_usd_log', 'goal_per_day_log', 'prep_time_ratio_log']

    for col in log_cols:
        if col in df.columns:
            c_min, c_median, c_max = df[col].min(), df[col].median(), df[col].max()
            # Un logaritmo sano non deve essere infinito o avere una dispersione folle
            print(f"   ✅ OK | {col.ljust(20)}: Min:{c_min:7.2f}, Med:{c_median:7.2f}, Max:{c_max:7.2f}")
        else:
            print(f"   ⚠️  MANCANTE | {col.ljust(20)}: Non trovata nel dataset!")

    # 3. VERIFICA RIDONDANZA (Pulizia colonne raw)
    print("\n🧹 VERIFICA PULIZIA COLONNE RAW")
    raw_cols = ['goal_per_day', 'prep_time_ratio', 'goal_usd']
    found_raw = [c for c in raw_cols if c in df.columns]
    if found_raw:
        print(f"   ⚠️  NOTA: Hai ancora le colonne raw {found_raw}. ")
        print("      Sarebbe meglio dropparle e tenere solo le versioni _log per il training.")
    else:
        print("   ✅ Perfetto: Le colonne raw esplosive sono state rimosse.")

    # 4. CONTROLLO INTEGRITÀ (I "Killer" dei Modelli)
    print("\n🩺 CONTROLLO VALORI CRITICI (NaN / Inf)")
    num_df = df.select_dtypes(include=np.number)
    inf_count = np.isinf(num_df).sum().sum()
    nan_count = num_df.isna().sum().sum()

    if inf_count == 0 and nan_count == 0:
        print("   ✅ Eccellente: Zero Infiniti e Zero Nulli.")
    else:
        if inf_count > 0:
            print(f"   ❌ ERRORE: Rilevati {inf_count} Infiniti! (Controlla log di zero)")
        if nan_count > 0:
            print(f"   ❌ ERRORE: Rilevati {nan_count} Nulli! (Controlla Imputazione)")

    # 5. TEST FINALE ANTI-GHOST (Excel vs Python)
    print("\n👻 TEST VERITÀ: Excel ti sta mentendo?")
    if trig_cols:
        ghosts = (df[trig_cols] < -1.0001).sum().sum()
        if ghosts == 0:
            print("   ✅ CONFERMATO: Python non vede valori < -1. Excel sta interpretando male i punti decimali.")
        else:
            print(f"   ❌ ALERT: Python vede {ghosts} valori fuori range. C'è un errore nel calcolo.")

    print("\n" + "=" * 80)
    print("🚀 CONCLUSIONE: Se tutti i check sono verdi, il dataset è PRONTO PER IL MODELING!")


# ESECUZIONE
# Inserisci il percorso corretto del file
audit_maniacale_v4_definitiva(r'C:\Users\polic\Desktop\KickProphet\app\data\processed\KICKSTARTER_TRAIN.csv')