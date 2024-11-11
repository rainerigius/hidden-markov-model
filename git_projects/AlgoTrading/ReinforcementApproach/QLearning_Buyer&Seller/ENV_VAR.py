APCA_API_BASE_URL='https://paper-api.alpaca.markets'
APCA_API_KEY_ID='yourkeyhere'
APCA_API_SECRET_KEY='yoursecretkeyhere'
AV_API_KEY_ID='yourkeyhere'
MARKET_OPEN = 570
MARKET_CLOSE = 960
TRADE_OPEN = MARKET_OPEN + 30
TRADE_CLOSE = MARKET_CLOSE - 15

# For data frame
OPEN = '1. open'
HIGH = '2. high'
LOW = '3. low'
CLOSE = '4. close'
VOLUME = '5. volume'

# For numpy array
TIME_INDEX = 0
OPEN_INDEX = 1
HIGH_INDEX = 2
LOW_INDEX = 3
CLOSE_INDEX = 4
VOLUME_INDEX = 5

# For histories
NUM_OPEN_HISTORY = 7
NUM_MOVING_AVG_HISTORY = 7

# For moving averages make sure highest value + history is < 30
MOVING_AVG_KS = [10,20]

# ACTION CONSTANTS
BUY = 0
SELL = 2
HOLD = 1

# Percent of current equity to use to buy shares (floored)
BUY_PERCENTAGE = 0.2

# Value to divide state space by (will always exceed any value)
MAX_VAL = 2000
