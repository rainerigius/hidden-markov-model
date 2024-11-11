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
        'KAMA_10'
    ]
    
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
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
        'KAMA_10'
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

    return model, scaler

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
        'KAMA_10'
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
        'KAMA_10'
    ]
    
    for state in range(model.n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[features].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

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
        'KAMA_10'
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
        'KAMA_10'
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
        'KAMA_10'
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

# Plot accuracy metrics
def plot_accuracy_metrics(silhouette_score_val, next_state_accuracy_val):
    print("Plotting accuracy metrics...")
    metrics = ['Silhouette Score', 'Next State Prediction Accuracy']
    scores = [silhouette_score_val, next_state_accuracy_val]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, scores, color=['blue', 'green'])
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
        'KAMA_10'
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

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, scores, color='cyan')
    plt.ylabel('Weight')
    plt.title('Feature Weights Based on Fisher\'s Score')
    plt.xticks(rotation=45)
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
    print("Feature weights plotted.")

# Main execution
def main():
    print("Starting main execution...")
    file_path = './BTC_1H.csv'
    data = load_and_preprocess_data(file_path)
    
    print("Training HMM model...")
    model, scaler = train_hmm(data)
    
    print("Cross-validating HMM model...")
    cross_validate_hmm(data)
    
    print("Predicting states...")
    states = predict_states(model, data, scaler)
    
    print("Analyzing states...")
    analyze_states(data, states, model)
    
    print("Calculating model accuracy...")
    silhouette = calculate_silhouette_score_func(data, states)
    next_state_accuracy = calculate_next_state_accuracy(states, model)
    
    print("Calculating feature weights...")
    fisher_scores = calculate_feature_weights(data, states)
    
    print("Printing transition matrix...")
    print("Transition Matrix:")
    print(model.transmat_)
    
    print("Plotting results...")
    plot_results(data, states, model)
    
    print("Plotting state probabilities...")
    plot_state_probabilities(model, data, scaler)
    
    print("Plotting model statistics...")
    plot_model_statistics(model)
    
    print("Plotting accuracy metrics...")
    plot_accuracy_metrics(silhouette, next_state_accuracy)
    
    print("Plotting feature weights...")
    plot_feature_weights(fisher_scores)
    
    # Show all plots at once
    print("Displaying all plots...")
    plt.show()
    
    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the trained HMM model and the scaler using joblib in the 'models' directory
    print("Saving the HMM model and scaler...")
    joblib.dump(model, './models/model_hmm_98%.joblib')          # Save the HMM model
    joblib.dump(scaler, './models/scaler_hmm_98%.joblib')      # Save the scaler
    print("Model and scaler saved successfully in the 'models/' directory.")
    
    print("HMM analysis completed.")

if __name__ == "__main__":
    main()


#https://www.youtube.com/watch?v=JwGfdnnyUAs&ab_channel=MoonDev
#2:18:19
