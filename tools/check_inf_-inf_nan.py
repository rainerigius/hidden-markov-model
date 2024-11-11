import pandas as pd
import numpy as np

def report_invalid_values(file_path):
    """
    Analizza un file CSV e stampa nel terminale le posizioni di valori NaN, inf e -inf.
    
    Parameters:
        file_path (str): Il percorso al file CSV da analizzare.
        
    Returns:
        None
    """
    try:
        # Carica i dati dal CSV
        print(f"Caricamento dei dati da '{file_path}'...")
        df = pd.read_csv(file_path)
        print("Dati caricati con successo.\n")
        
        # Seleziona solo le colonne numeriche
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("Nessuna colonna numerica trovata nel dataset.")
            return
        
        # Inizializza una variabile per tracciare se sono stati trovati valori invalidi
        found_invalid = False
        
        # Itera su ogni colonna numerica del DataFrame
        for col in numeric_cols:
            # Crea una maschera per NaN, inf e -inf
            mask_nan = df[col].isna()
            mask_inf = np.isinf(df[col])
            mask_invalid = mask_nan | mask_inf
            
            # Se ci sono valori invalidi nella colonna, stampali
            if mask_invalid.any():
                invalid_rows = df.index[mask_invalid].tolist()
                for row in invalid_rows:
                    value = df.at[row, col]
                    print(f"Riga {row + 2}, Colonna '{col}': Valore = {value}")
                    found_invalid = True
        
        if not found_invalid:
            print("Nessun valore NaN, inf o -inf trovato nel dataset.")
        else:
            print("\nAnalisi completata. Sono stati trovati valori invalidi come sopra elencati.")
    
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
    except pd.errors.EmptyDataError:
        print("Errore: Il file CSV è vuoto.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    # Specifica il percorso al tuo file CSV
    input_file = './gspc.csv'  # Modifica questo percorso se necessario
    
    # Chiama la funzione per reportare valori invalidi
    report_invalid_values(input_file)
