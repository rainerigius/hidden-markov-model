import yfinance as yf
import pandas as pd

# Definisci il ticker e scarica i dati storici massimi da Yahoo Finance
ticker = '^GSPC'
data = yf.download(ticker, period="max")

# Seleziona e rinomina le colonne
cleaned_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
cleaned_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# Formatta la colonna 'date' nel formato YYYY-MM-DD
cleaned_data['date'] = cleaned_data['date'].dt.strftime('%Y-%m-%d')

# Salva i dati puliti in un file CSV
cleaned_data.to_csv('./gspc.csv', index=False)

print("Dati scaricati e salvati come 'gspc.csv' con la colonna 'date' nel formato YYYY-MM-DD.")
