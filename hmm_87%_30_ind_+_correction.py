import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit
import time
import pandas_ta as ta
import talib
import joblib  # Added import

# Set this before importing libraries that use joblib/loky
os.environ['LOKY_MAX_CPU_COUNT'] = '6'  # Replace '6' with the desired number of cores

print("Starting Bitcoin HMM analysis...")

# Load and process data
def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure required columns are present
    required_columns = ['open','high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")
    
    # Data Quality Checks
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")
    
    print("Creating datetime index...")
    # Addressing the FutureWarning by using lowercase 'h'
    df.index = pd.date_range(start='2022-01-01', periods=len(df), freq='h')
    
    print("Calculating technical indicators...")
    # Calculate indicators from pandas_ta
    df['ADX_14'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    donchian = ta.donchian(df['high'], df['low'], length=20)
    
    # Verify Donchian Channel columns
    print("Donchian Channel columns available:", donchian.columns.tolist())
    
    # Correct column names based on pandas_ta's output
    df['DCH_20'] = donchian['DCU_20_20']  # Donchian Channel Upper
    df['DCL_20_20'] = donchian['DCL_20_20']  # Donchian Channel Lower
    
    df['LINREG_10'] = ta.linreg(df['close'], length=10)
    macd = ta.macd(df['close'])
    df['MACD_12_26_9'] = macd['MACD_12_26_9']
    
    # Attempt to calculate True Range using pandas_ta
    try:
        df['TR'] = ta.trange(df['high'], df['low'], df['close'])
    except AttributeError:
        print("pandas_ta.trange not found. Calculating True Range manually.")
        df['prev_close'] = df['close'].shift(1)
        df['TR'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['prev_close']))
        df['TR'] = np.maximum(df['TR'], np.abs(df['low'] - df['prev_close']))
        df.drop('prev_close', axis=1, inplace=True)
    
    # Calculate indicators from ta-lib
    stochrsi = talib.STOCHRSI(df['close'].values, timeperiod=14)
    df['STOCHRSI_K_14'] = stochrsi[0]
    df['STOCHRSI_D_14'] = stochrsi[1]
    df['KAMA_10'] = talib.KAMA(df['close'].values, timeperiod=10)
    
    # === Additional Technical Indicators ===
    
    ## 1. Volume-Based Indicators
    print("Calculating Volume-Based Indicators...")
    # On-Balance Volume (OBV)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    
    # Chaikin Money Flow (CMF)
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    
    ## 2. Additional Momentum Indicators
    print("Calculating Additional Momentum Indicators...")
    # Relative Strength Index (RSI)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    
    # Williams %R
    df['Williams_%R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    
    ## 3. Additional Volatility Indicators
    print("Calculating Additional Volatility Indicators...")
    # Bollinger Bands
    bbands = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = bbands[0]
    df['BB_middle'] = bbands[1]
    df['BB_lower'] = bbands[2]
    
    # Keltner Channels
    df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
    df['ATR_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
    df['KC_upper'] = df['EMA_20'] + (2 * df['ATR_20'])
    df['KC_lower'] = df['EMA_20'] - (2 * df['ATR_20'])
    
    ## 4. Trend Indicators
    print("Calculating Trend Indicators...")
    # Parabolic SAR
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    
    # Ichimoku Cloud Components using pandas_ta
    ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
    
    # Debugging: Print Ichimoku columns
    print("Ichimoku calculation completed. Columns available:")
    if isinstance(ichimoku, tuple):
        ichimoku_df = ichimoku[0]
    else:
        ichimoku_df = ichimoku
    print(ichimoku_df.columns.tolist())
    
    # Concatenate Ichimoku indicators to the main DataFrame
    df = pd.concat([df, ichimoku_df], axis=1)
    
    ## 5. Additional Indicators
    print("Calculating Additional Indicators...")
    # ADX Variations
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Candlestick Patterns
    df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    # Add other candlestick patterns as needed
    # Example:
    # df['CDLHARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    
    # Fibonacci Retracement Levels
    print("Calculating Fibonacci Retracement Levels...")
    window = 50  # Number of periods to look back for high and low
    recent_high = df['high'].rolling(window=window, min_periods=1).max()
    recent_low = df['low'].rolling(window=window, min_periods=1).min()
    df['Fib_23.6'] = recent_low + 0.236 * (recent_high - recent_low)
    df['Fib_38.2'] = recent_low + 0.382 * (recent_high - recent_low)
    df['Fib_50.0'] = recent_low + 0.5 * (recent_high - recent_low)
    df['Fib_61.8'] = recent_low + 0.618 * (recent_high - recent_low)
    df['Fib_76.4'] = recent_low + 0.764 * (recent_high - recent_low)
    
    # === End of Additional Indicators ===
    
    # Handle infinite values
    print("Replacing infinite values with NaN...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print("Dropping NaN values...")
    df.dropna(inplace=True)
    
    # Report data loss
    final_shape = df.shape
    data_loss = (initial_shape[0] - final_shape[0]) / initial_shape[0] * 100
    print(f"Data preprocessed. Final shape {final_shape}. Data loss: {data_loss:.2f}%")
    
    # Verify all required features are present
    required_features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        # Ichimoku Components (updated suffixes based on default periods)
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        print("Missing features after preprocessing:", missing_features)
        raise ValueError(f"Missing features after preprocessing: {missing_features}")
    
    return df

# Train HMM with additional metrics
def train_hmm(data, n_components=3):
    print(f"Training HMM with {n_components} components...")
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values

    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        verbose=False
    )
    
    start_time = time.time()
    model.fit(X_scaled)
    end_time = time.time()
    
    # Check convergence
    if hasattr(model.monitor_, 'converged'):
        converged = model.monitor_.converged
        iterations = model.monitor_.iter
        print(f"HMM training completed in {end_time - start_time:.2f} seconds.")
        print(f"Converged: {converged} after {iterations} iterations.")
    else:
        print(f"HMM training completed in {end_time - start_time:.2f} seconds.")
        print("Convergence information not available.")
    
    # Log-Likelihood, AIC, BIC
    log_likelihood = model.score(X_scaled)
    n_features = X_scaled.shape[1]
    n_parameters = (n_components - 1) + n_components * (n_features + n_features * (n_features + 1) / 2)
    aic = -2 * log_likelihood + 2 * n_parameters
    bic = -2 * log_likelihood + n_parameters * np.log(X_scaled.shape[0])
    print(f"Log-Likelihood: {log_likelihood:.2f}")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")

    return model, scaler, X_scaled

# Predict states
def predict_states(model, data, scaler):
    print("Predicting states...")
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    print(f"States predicted. Unique states: {np.unique(states)}")
    return states

# Analyze states
def analyze_states(data, states, model):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    
    for state in range(model.n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[features].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

# Predict next state
def predict_next_state(model, current_state):
    return np.argmax(model.transmat_[current_state])

# Save state changes
def save_state_changes(states, data, state_names, output_file):
    state_changes = []
    current_state = states[0]
    start_time = data.index[0]
    for i, state in enumerate(states[1:], 1):
        if state != current_state:
            end_time = data.index[i-1]
            state_changes.append((start_time, end_time, current_state))
            current_state = state
            start_time = data.index[i]
    
    # Append the last state
    state_changes.append((start_time, data.index[-1], current_state))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Start Time,End Time,State,State Name\n")
        for start, end, state in state_changes:
            f.write(f"{start},{end},{state},{state_names[state]}\n")

    print(f"State changes have been saved to {output_file}")

# Plot results
def plot_results(data, states, model):
    print("Plotting results...")
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['close'], label='Close Price', color='black')
    plt.title('Price and HMM States')
    plt.ylabel('Price')
    plt.xlabel('Date')

    # Updated to use plt.get_cmap to avoid deprecation warning
    colors = plt.get_cmap('Set1', model.n_components)
    for state in range(model.n_components):
        mask = (states == state)
        plt.fill_between(
            data.index,
            data['close'].min(),
            data['close'].max(),
            where=mask,
            alpha=0.1,
            color=colors(state),
            label=f'State {state}'
        )

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking
    plt.pause(0.001)  # Allow GUI events to process
    print("Plotting completed.")

# Cross-Validation
def cross_validate_hmm(data, n_components=3, n_splits=5):
    print("Starting cross-validation...")
    
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    log_likelihoods = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        print(f"Fold {fold+1}")
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        model_cv = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model_cv.fit(X_train)
        log_likelihood = model_cv.score(X_test)
        log_likelihoods.append(log_likelihood)
        print(f"Log-Likelihood on test set: {log_likelihood:.2f}")

    avg_log_likelihood = np.mean(log_likelihoods)
    print(f"Average Log-Likelihood over {n_splits} folds: {avg_log_likelihood:.2f}")
    return avg_log_likelihood

# Visualize posterior probabilities
def plot_state_probabilities(model, data, scaler):
    print("Plotting state probabilities...")
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values
    X_scaled = scaler.transform(X)
    state_probs = model.predict_proba(X_scaled)

    plt.figure(figsize=(15, 7))
    # Updated to use plt.get_cmap to avoid deprecation warning
    colors = plt.get_cmap('Set1', model.n_components)
    for i in range(model.n_components):
        plt.plot(data.index, state_probs[:, i], label=f'State {i}', color=colors(i))
    plt.title('State Posterior Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking
    plt.pause(0.001)  # Allow GUI events to process
    print("State probabilities plotted.")

# Plot model statistics
def plot_model_statistics(model):
    print("Plotting model statistics...")
    fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size
    ax.axis('off')  # Hide the axes

    text = ""
    text += "Transition Matrix:\n"
    text += f"{model.transmat_}\n\n"

    text += "Means of Each State:\n"
    for i in range(model.n_components):
        text += f"State {i}: {model.means_[i]}\n"
    text += "\n"

    text += "Covariances of Each State:\n"
    for i in range(model.n_components):
        text += f"State {i}:\n{model.covars_[i]}\n\n"

    # Display the text with smaller font size
    ax.text(0.01, 0.99, text, fontsize=10, verticalalignment='top', transform=ax.transAxes)
    plt.title('Model Statistics')
    
    # Adjust layout to accommodate text
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking
    plt.pause(0.001)  # Allow GUI events to process
    print("Model statistics plotted.")

# Calculate model accuracy using Silhouette Score
def calculate_silhouette_score_func(data, states):
    print("Calculating Silhouette Score...")
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    score = silhouette_score(X_scaled, states)
    print(f"Silhouette Score: {score:.4f}")
    return score

# Calculate next state prediction accuracy
def calculate_next_state_accuracy(states, model):
    print("Calculating next state prediction accuracy...")
    transmat = model.transmat_
    correct_predictions = 0
    total_predictions = len(states) - 1  # Since we cannot predict the last state

    for t in range(len(states) - 1):
        current_state = states[t]
        actual_next_state = states[t + 1]
        # Predicted next state is the one with highest transition probability from current_state
        predicted_next_state = np.argmax(transmat[current_state])
        if predicted_next_state == actual_next_state:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Next state prediction accuracy: {accuracy:.4f}")
    return accuracy

# Plot accuracy metrics including overfitting metric
def plot_accuracy_metrics(silhouette_score_val, next_state_accuracy_val, overfitting_metric_val):
    print("Plotting accuracy metrics...")
    metrics = ['Silhouette Score', 'Next State Prediction Accuracy', 'Overfitting Metric']
    scores = [silhouette_score_val, next_state_accuracy_val, overfitting_metric_val]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, scores, color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model Accuracy Metrics')
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + 0.02,
            f"{yval:.4f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking
    plt.pause(0.001)  # Allow GUI events to process
    print("Accuracy metrics plotted.")

# Calculate feature weights using Fisher's Score
def calculate_feature_weights(data, states):
    print("Calculating feature weights using Fisher's Score...")
    features = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df = pd.DataFrame(X_scaled, columns=features)
    df['State'] = states

    overall_mean = df[features].mean()

    fisher_scores = {}
    for feature in features:
        numerator = 0
        denominator = 0
        for state in df['State'].unique():
            state_data = df[df['State'] == state]
            n_state = len(state_data)
            state_mean = state_data[feature].mean()
            state_variance = state_data[feature].var()
            numerator += n_state * (state_mean - overall_mean[feature]) ** 2
            denominator += n_state * state_variance
        fisher_scores[feature] = numerator / denominator if denominator != 0 else 0

    # Normalize the scores to sum to 1
    total_score = sum(fisher_scores.values())
    for feature in fisher_scores:
        fisher_scores[feature] /= total_score if total_score != 0 else 1

    print("Feature Weights (Normalized Fisher's Score):")
    for feature, score in fisher_scores.items():
        print(f"{feature}: {score:.4f}")
    
    return fisher_scores

# Plot feature weights
def plot_feature_weights(fisher_scores):
    print("Plotting feature weights...")
    features = list(fisher_scores.keys())
    scores = list(fisher_scores.values())

    plt.figure(figsize=(12, 8))
    bars = plt.bar(features, scores, color='cyan')
    plt.ylabel('Weight')
    plt.title('Feature Weights Based on Fisher\'s Score')
    plt.xticks(rotation=90)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + 0.005,
            f"{yval:.4f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking
    plt.pause(0.001)  # Allow GUI events to process
    print("Feature weights plotted.")

# Calculate prediction accuracy
def calculate_prediction_accuracy(true_states, predicted_states):
    if len(true_states) <= 1 or len(predicted_states) <= 1:
        print("Not enough data to calculate prediction accuracy.")
        return None
    return np.mean(np.array(true_states[1:]) == np.array(predicted_states[:-1]))

# Calculate BIC
def calculate_bic(model, X):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    n_params = (model.n_components - 1) + model.n_components * (n_features + n_features * (n_features + 1) / 2)
    bic = -2 * model.score(X) + n_params * np.log(n_samples)
    return bic

# Time Series Cross-Validation
def time_series_cv(X, n_components=3, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"Training fold {fold}...")
        X_train, X_test = X[train_index], X[test_index]
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42, verbose=False)
        model.fit(X_train)
        score = model.score(X_test)
        print(f"Fold {fold} score: {score:.2f}")
        scores.append(score)

    return np.mean(scores), np.std(scores)

# Analyze feature importance
def analyze_feature_importance(model, feature_names):
    importance = np.abs(model.means_).sum(axis=0)
    importance /= importance.sum()
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")

# Main execution
def main():
    print("Starting main execution...")
    file_path = './btc.csv'
    data = load_and_preprocess_data(file_path)
    
    print("Training HMM model...")
    model, scaler, X_scaled = train_hmm(data)
    
    print("Cross-validating HMM model...")
    cross_val_log_likelihood = cross_validate_hmm(data)
    
    print("Predicting states...")
    states = predict_states(model, data, scaler)
    
    state_names = [
        "Bullish Trend",
        "Bearish Trend",
        "Sideways/Consolidation"
    ]
    
    output_file = 'data/bitcoin_state_changes.csv'
    save_state_changes(states, data, state_names, output_file)
    
    print("Analyzing states...")
    analyze_states(data, states, model)
    
    print("Calculating model accuracy...")
    silhouette = calculate_silhouette_score_func(data, states)
    next_state_accuracy = calculate_next_state_accuracy(states, model)
    
    # Calculate overfitting metric
    print("Calculating overfitting metric...")
    training_log_likelihood = model.score(X_scaled)
    cross_val_log_likelihood = cross_val_log_likelihood  # Already obtained from cross_validate_hmm
    overfitting_metric = 1 - (abs(training_log_likelihood - cross_val_log_likelihood) / abs(training_log_likelihood))
    overfitting_metric = min(1, max(0, overfitting_metric))  # Clamp between 0 and 1
    print(f"Overfitting Metric: {overfitting_metric:.4f}")
    
    # Interpretation based on overfitting_metric
    if overfitting_metric > 0.8:
        print("IT DOES NOT OVERFIT")
    else:
        print("IT OVERFITS")
    
    print("Calculating feature weights...")
    fisher_scores = calculate_feature_weights(data, states)
    
    print("Printing transition matrix...")
    print("Transition Matrix:")
    print(model.transmat_)
    
    print("\nPrinting means and covariances of each state...")
    for i in range(model.n_components):
        print(f"State {i} ({state_names[i]}):")
        print("Mean:", model.means_[i])
        print("Covariance:", model.covars_[i])
        print()
    
    print("Plotting results...")
    plot_results(data, states, model)
    
    print("Plotting state probabilities...")
    plot_state_probabilities(model, data, scaler)
    
    print("Plotting model statistics...")
    plot_model_statistics(model)
    
    print("Plotting accuracy metrics...")
    plot_accuracy_metrics(silhouette, next_state_accuracy, overfitting_metric)
    
    print("Plotting feature weights...")
    plot_feature_weights(fisher_scores)
    
    # Continue with further processing or wait until plots are closed
    # For demonstration, we'll keep the script running until all plots are closed
    print("Displaying all plots. Close the plot windows to end the script.")

    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the trained HMM model and the scaler using joblib in the 'models' directory
    print("Saving the HMM model and scaler...")
    joblib.dump(model, './models/model_hmm_85%_30ind_updated.joblib')          # Save the HMM model
    joblib.dump(scaler, './models/scaler_hmm_85%_30ind_updated.joblib')      # Save the scaler
    print("Model and scaler saved successfully in the 'models/' directory.")
    
    print("HMM analysis completed.")
    
    print("Predicting next states based on current states...")
    next_state_predictions = [predict_next_state(model, state) for state in states]

    accuracy = calculate_prediction_accuracy(states, next_state_predictions)
    if accuracy is not None:
        print(f"State Prediction Accuracy: {accuracy:.2f}")
    
    log_likelihood = model.score(X_scaled)
    print(f"Log-Likelihood: {log_likelihood:.2f}")
    
    bic = calculate_bic(model, X_scaled)
    print(f"BIC: {bic:.2f}")
    
    print("Performing Time Series Cross-Validation...")
    cv_mean, cv_std = time_series_cv(X_scaled)
    print(f"Cross-Validation Score: {cv_mean:.2f} (+/- {cv_std:.2f})")
    
    print("\nFeature Importance Analysis:")
    feature_names = [
        'ADX_14',
        'ATR_14',
        'DCH_20',
        'DCL_20_20',
        'LINREG_10',
        'MACD_12_26_9',
        'TR',
        'STOCHRSI_K_14',
        'KAMA_10',
        # === Additional Features ===
        'OBV',
        'CMF',
        'RSI_14',
        'Williams_%R',
        'BB_upper',
        'BB_middle',
        'BB_lower',
        'KC_upper',
        'KC_lower',
        'SAR',
        'ISA_9',
        'ISB_26',
        'ITS_9',
        'IKS_26',
        'ICS_26',
        'PLUS_DI',
        'MINUS_DI',
        'CDLENGULFING',
        'CDLDOJI',
        'Fib_23.6',
        'Fib_38.2',
        'Fib_50.0',
        'Fib_61.8',
        'Fib_76.4'
        # === End of Additional Features ===
    ]
    analyze_feature_importance(model, feature_names)
    
    # Ensure the 'trained_models' directory exists
    os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    main()

#https://www.youtube.com/watch?v=JwGfdnnyUAs&ab_channel=MoonDev
#2:18:19
