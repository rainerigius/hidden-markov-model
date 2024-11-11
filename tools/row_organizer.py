import pandas as pd
import os

def reverse_csv_rows(input_file, output_file=None, overwrite=False):
    """
    Inverte l'ordine delle righe di un file CSV.

    Parameters:
        input_file (str): Percorso al file CSV di input.
        output_file (str, optional): Percorso al file CSV di output. Se non specificato e overwrite=False, viene creato un nuovo file con '_reversed' aggiunto al nome.
        overwrite (bool, optional): Se True, sovrascrive il file di input. Defaults to False.

    Returns:
        None
    """
    try:
        # Verifica se il file di input esiste
        if not os.path.isfile(input_file):
            print(f"Errore: Il file '{input_file}' non esiste.")
            return

        # Leggi il file CSV
        print(f"Caricamento del file CSV: '{input_file}'...")
        df = pd.read_csv(input_file)
        print("File caricato con successo.")

        # Inverti l'ordine delle righe
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        print("Ordine delle righe invertito.")

        # Determina il percorso del file di output
        if overwrite:
            save_path = input_file
            print("Sovrascrittura del file originale.")
        else:
            if output_file:
                save_path = output_file
            else:
                base, ext = os.path.splitext(input_file)
                save_path = f"{base}_reversed{ext}"
            print(f"Salvataggio del file invertito come: '{save_path}'")

        # Salva il DataFrame invertito nel file CSV
        df_reversed.to_csv(save_path, index=False)
        print("File salvato con successo.")

    except pd.errors.EmptyDataError:
        print("Errore: Il file CSV è vuoto.")
    except pd.errors.ParserError:
        print("Errore: Si è verificato un problema durante l'analisi del file CSV.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

def main():
    print("=== Invertitore di Righe CSV ===")

    # Richiedi all'utente il percorso del file CSV
    input_file = input("Inserisci il percorso del file CSV da invertire: ").strip()

    if not input_file:
        print("Errore: Il percorso del file non può essere vuoto.")
        return

    # Chiedi all'utente se desidera sovrascrivere il file originale
    while True:
        overwrite_input = input("Vuoi sovrascrivere il file originale? (sì/no): ").strip().lower()
        if overwrite_input in ['sì', 'si', 's']:
            overwrite = True
            output_file = None
            break
        elif overwrite_input in ['no', 'n']:
            overwrite = False
            # Richiedi all'utente il percorso del file di output
            output_file = input("Inserisci il percorso del file CSV di output (lascia vuoto per aggiungere '_reversed' al nome originale): ").strip()
            if not output_file:
                output_file = None
            break
        else:
            print("Input non valido. Per favore, rispondi con 'sì' o 'no'.")

    # Chiama la funzione per invertire le righe
    reverse_csv_rows(input_file, output_file, overwrite)

if __name__ == "__main__":
    main()
