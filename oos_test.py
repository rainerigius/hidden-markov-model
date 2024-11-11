import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import joblib

# Load the saved models and scalers
model_volume = joblib.load("./models/model_hmm_98%.joblib")  # Volume_Change, BB_Width, Volatility
scaler_volume = joblib.load("./models/scaler_hmm_98%.joblib")
model_linreg = joblib.load("./models/model_hmm_88%.joblib")  # LINREG, MACD
scaler_linreg = joblib.load("./models/scaler_hmm_88%.joblib")

def preprocess_data(df):
    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=24).std()
    df['volume_change'] = df['volume'].pct_change()
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
    df['linreg'] = ta.linreg(df['close'], length=14)['SLOPE_14']
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# Load and preprocess out-of-sample data
oos_data = pd.read_csv('/Users/md/Dropbox/dev/github/hidden-markov-model-4-trading/oos_data.csv')
oos_data['datetime'] = pd.to_datetime(oos_data['datetime'])
oos_data.set_index('datetime', inplace=True)
oos_data = preprocess_data(oos_data)

# Evaluate Volume_Change model
X_volume = oos_data[['volume_change', 'bb_width', 'volatility']].values
X_volume_scaled = scaler_volume.transform(X_volume)
states_volume = model_volume.predict(X_volume_scaled)
log_likelihood_volume = model_volume.score(X_volume_scaled)

# Evaluate LINREG model
X_linreg = oos_data[['linreg', 'macd', 'true_range']].values
X_linreg_scaled = scaler_linreg.transform(X_linreg)
states_linreg = model_linreg.predict(X_linreg_scaled)
log_likelihood_linreg = model_linreg.score(X_linreg_scaled)

# Print results
print("Model Volume (Volume_Change, BB_Width, Volatility) Log-Likelihood:", log_likelihood_volume)
print("Model LINREG (LINREG, MACD, TRUE_RANGE) Log-Likelihood:", log_likelihood_linreg)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

ax1.plot(oos_data.index, oos_data['close'])
ax1.set_title('Bitcoin Price and HMM States (Out-of-Sample)')
ax1.set_ylabel('Price')

colors = plt.cm.rainbow(np.linspace(0, 1, 7))

for state in range(7):
    ax2.fill_between(oos_data.index, 0, 1, where=(states_volume == state),
                     alpha=0.3, color=colors[state], label=f'State {state}')
    ax3.fill_between(oos_data.index, 0, 1, where=(states_linreg == state),
                     alpha=0.3, color=colors[state], label=f'State {state}')

ax2.set_title('Model Volume States (Volume_Change, BB_Width, Volatility)')
ax2.set_ylabel('State')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax3.set_title('Model LINREG States (LINREG, MACD, TRUE_RANGE)')
ax3.set_ylabel('State')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
