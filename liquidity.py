import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def liquidity_heatmap(csv_path):
    """
    Reads market data from a CSV file and plots the Liquidity Heatmap indicator.
    
    Parameters:
    - csv_path: str, path to the CSV file containing 'date', 'open', 'high', 'low', 'close', 'volume'
    """
    
    # Configuration Parameters
    mode = "HD"  # Options: "Normal", "HD"
    side = "Both"  # Options: "Both", "Long", "Short"
    LV = 300  # Leverage factor
    long_color = "#cc00cc"  # Long liquidation color
    short_color = "#ffff00"  # Short liquidation color
    primary_color = "#ffffff"  # Primary color for max bubble

    # Load data
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate SMA for volume to determine significant volume spikes
    df['sma_volume'] = df['volume'].rolling(window=14).mean()
    df['fT'] = df['volume'] > df['sma_volume']

    # Initialize columns for liquidation levels and bubble properties
    df['long_liquidation'] = np.nan
    df['short_liquidation'] = np.nan
    df['bubble_size'] = np.nan
    df['bubble_color'] = None

    # Process each row to determine liquidation levels
    for i in range(1, len(df)):
        if df.loc[i, 'fT']:
            # Set liquidation level based on mode
            volume_effect = df.loc[i, 'volume'] if mode == "HD" else df.loc[i, 'volume'] - df.loc[i-1, 'volume']

            # Long liquidation level
            if side in ["Both", "Long"] and df.loc[i, 'close'] > df.loc[i, 'open']:
                df.loc[i, 'long_liquidation'] = df.loc[i, 'low'] * (1 - 1.0 / LV)
                # Determine bubble size and color
                if volume_effect == df['volume'].max():
                    df.loc[i, 'bubble_size'] = 60
                    df.loc[i, 'bubble_color'] = primary_color
                elif volume_effect > df['volume'].mean():
                    df.loc[i, 'bubble_size'] = 40
                    df.loc[i, 'bubble_color'] = long_color
                else:
                    df.loc[i, 'bubble_size'] = 20
                    df.loc[i, 'bubble_color'] = long_color

            # Short liquidation level
            if side in ["Both", "Short"] and df.loc[i, 'close'] < df.loc[i, 'open']:
                df.loc[i, 'short_liquidation'] = df.loc[i, 'high'] * (1 + 1.0 / LV)
                # Determine bubble size and color
                if volume_effect == df['volume'].max():
                    df.loc[i, 'bubble_size'] = 60
                    df.loc[i, 'bubble_color'] = primary_color
                elif volume_effect > df['volume'].mean():
                    df.loc[i, 'bubble_size'] = 40
                    df.loc[i, 'bubble_color'] = short_color
                else:
                    df.loc[i, 'bubble_size'] = 20
                    df.loc[i, 'bubble_color'] = short_color

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot Close Price
    ax.plot(df['date'], df['close'], label='Close Price', color='blue', linewidth=1)

    # Plot Long Liquidation Levels with bubbles
    if side in ["Both", "Long"]:
        long_bubbles = df.dropna(subset=['long_liquidation'])
        ax.scatter(long_bubbles['date'], long_bubbles['long_liquidation'], 
                   s=long_bubbles['bubble_size'], color=long_bubbles['bubble_color'], 
                   label='Long Liquidation Bubbles', alpha=0.7)

    # Plot Short Liquidation Levels with bubbles
    if side in ["Both", "Short"]:
        short_bubbles = df.dropna(subset=['short_liquidation'])
        ax.scatter(short_bubbles['date'], short_bubbles['short_liquidation'], 
                   s=short_bubbles['bubble_size'], color=short_bubbles['bubble_color'], 
                   label='Short Liquidation Bubbles', alpha=0.7)

    # Add max liquidation levels as horizontal lines
    if not long_bubbles.empty:
        ax.hlines(y=long_bubbles['long_liquidation'].max(), xmin=long_bubbles['date'].min(), xmax=long_bubbles['date'].max(),
                  color=long_color, linestyle='--', linewidth=1, alpha=0.6)
    if not short_bubbles.empty:
        ax.hlines(y=short_bubbles['short_liquidation'].max(), xmin=short_bubbles['date'].min(), xmax=short_bubbles['date'].max(),
                  color=short_color, linestyle='--', linewidth=1, alpha=0.6)

    # Format the plot
    ax.set_title('Liquidity Heatmap [StratifyTrade]')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    # Improve date formatting on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.show()

# Usage Example
# liquidity_heatmap('path_to_your_market_data.csv')
liquidity_heatmap('BTC_1H.csv')