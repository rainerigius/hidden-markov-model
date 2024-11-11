# API
APCA_API_BASE_URL='https://paper-api.alpaca.markets'
APCA_API_KEY_ID='yourkeyhere'
APCA_API_SECRET_KEY='yoursecretkeyhere'
AV_API_KEY_ID='yourkeyhere'
APCA_MAX_CALLS = 200
AV_MAX_CALLS = 5

# Indices for data
OPEN_INDEX = 0
HIGH_INDEX = 1
LOW_INDEX = 2
CLOSE_INDEX = 3
VOLUME_INDEX = 4
METRICS = ['Open','High','Low','Close','Volume']

# For historical data (Set end to something very close to current day, like a week before)
DATE_START = '2016-01-01'
DATE_END = '2020-08-05'

# Default tolerances
DEFAULT_TOLS = (0.01,-0.01)

# All the file names
BUY_FILE = 'BuyStocks.txt'
SELL_FILE = 'SellStocks.txt'
OWNED_FILE = 'OwnedStocks.txt'
LOG_FILE = 'log.txt'
PL_FILE = 'profit_loss.txt'
