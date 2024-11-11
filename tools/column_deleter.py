import os
import glob
import pandas as pd

def list_csv_files(folder_path):
    """
    Elenca tutti i file CSV nella cartella specificata.

    Parameters:
        folder_path (str): Percorso alla cartella.

    Returns:
        list: Lista dei percorsi dei file CSV.
    """
    pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(pattern)
    return csv_files

def display_csv_files(csv_files):
    """
    Mostra una lista numerata dei file CSV.

    Parameters:
        csv_files (list): Lista dei percorsi dei file CSV.
    """
    if not csv_files:
        print("Nessun file CSV trovato nella cartella specificata.")
        return

    print("\nTrovati i seguenti file CSV:")
    for idx, file in enumerate(csv_files, start=1):
        print(f"{idx}. {os.path.basename(file)}")

def get_user_selection(total_files):
    """
    Chiede all'utente di selezionare i file CSV su cui operare.

    Parameters:
        total_files (int): Numero totale di file CSV.

    Returns:
        list: Lista degli indici dei file selezionati.
    """
    while True:
        selection = input("\nInserisci i numeri dei file CSV su cui vuoi operare (separati da virgole), oppure 'all' per selezionare tutti i file: ").strip()
        if selection.lower() == 'all':
            return list(range(1, total_files + 1))
        else:
            try:
                selected_indices = [int(num.strip()) for num in selection.split(",")]
                if all(1 <= idx <= total_files for idx in selected_indices):
                    return selected_indices
                else:
                    print(f"Per favore, inserisci numeri tra 1 e {total_files}.")
            except ValueError:
                print("Input non valido. Per favore, inserisci numeri separati da virgole oppure 'all'.")

def get_columns_to_delete(columns):
    """
    Chiede all'utente di selezionare le colonne da eliminare.

    Parameters:
        columns (list): Lista dei nomi delle colonne.

    Returns:
        list: Lista delle colonne da eliminare.
    """
    print("\nColonne disponibili:")
    for idx, col in enumerate(columns, start=1):
        print(f"{idx}. {col}")

    while True:
        selection = input("\nInserisci i numeri delle colonne che vuoi eliminare (separati da virgole), oppure 'none' per saltare: ").strip()
        if selection.lower() == 'none' or selection == '':
            return []
        else:
            try:
                selected_indices = [int(num.strip()) for num in selection.split(",")]
                if all(1 <= idx <= len(columns) for idx in selected_indices):
                    selected_columns = [columns[idx - 1] for idx in selected_indices]
                    return selected_columns
                else:
                    print(f"Per favore, inserisci numeri tra 1 e {len(columns)}.")
            except ValueError:
                print("Input non valido. Per favore, inserisci numeri separati da virgole oppure 'none'.")

def delete_columns_from_csv(file_path, columns_to_delete):
    """
    Elimina le colonne specificate da un file CSV senza alterare l'ordine delle righe.

    Parameters:
        file_path (str): Percorso al file CSV.
        columns_to_delete (list): Lista delle colonne da eliminare.
    """
    try:
        # Leggi il file CSV preservando l'ordine delle righe
        df = pd.read_csv(file_path, dtype=str)
        
        # Memorizza l'ordine originale delle righe
        original_order = df.index.tolist()
        
        print(f"\nElaborazione del file: {os.path.basename(file_path)}")

        # Verifica se le colonne da eliminare esistono nel DataFrame
        existing_columns = [col for col in columns_to_delete if col in df.columns]
        if not existing_columns:
            print("Nessuna colonna corrispondente trovata per l'eliminazione.")
            return

        # Mostra le colonne da eliminare
        print(f"Eliminazione delle colonne: {', '.join(existing_columns)}")

        # Elimina le colonne selezionate
        df.drop(columns=existing_columns, inplace=True)

        # Ripristina l'ordine originale delle righe
        df = df.reindex(original_order)

        # Crea un backup del file originale se non esiste già
        backup_path = file_path + ".backup"
        if not os.path.exists(backup_path):
            df_original = pd.read_csv(file_path)
            df_original.to_csv(backup_path, index=False)
            print(f"Backup del file originale creato in: {backup_path}")

        # Salva il DataFrame modificato nel file CSV originale
        df.to_csv(file_path, index=False)
        print("Colonne eliminate e file aggiornato con successo.")

    except Exception as e:
        print(f"Si è verificato un errore durante l'elaborazione di '{file_path}': {e}")

def main():
    print("=== Gestore delle Colonne CSV ===")

    # Specifica il percorso della cartella
    folder_path = input("Inserisci il percorso della cartella contenente i file CSV (lascia vuoto per la cartella corrente): ").strip()
    if not folder_path:
        folder_path = os.getcwd()

    if not os.path.isdir(folder_path):
        print("La cartella specificata non esiste.")
        return

    # Elenca tutti i file CSV
    csv_files = list_csv_files(folder_path)
    if not csv_files:
        print("Nessun file CSV trovato nella cartella specificata.")
        return

    # Mostra i file CSV
    display_csv_files(csv_files)

    # Ottieni la selezione dell'utente
    selected_indices = get_user_selection(len(csv_files))
    selected_files = [csv_files[idx - 1] for idx in selected_indices]

    # Itera sui file selezionati
    for file_path in selected_files:
        try:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
        except Exception as e:
            print(f"Impossibile leggere '{os.path.basename(file_path)}': {e}")
            continue

        # Ottieni le colonne da eliminare
        columns_to_delete = get_columns_to_delete(columns)

        if columns_to_delete:
            delete_columns_from_csv(file_path, columns_to_delete)
        else:
            print("Nessuna colonna selezionata per l'eliminazione. Saltando questo file.")

    print("\n=== Operazione Completata ===")

if __name__ == "__main__":
    main()
