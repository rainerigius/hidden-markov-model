from IMPORTS import *
from ENV_VAR import *

# *** Any function that calls the API should wait 1 minute after finished ***

# Gets current time hh:mm
def get_time(asint=False):
    if asint:
        dtime = datetime.datetime.now()
        return dtime.hour*60 + dtime.minute
    return time.strftime('%H:%M')

# Sleep for either specified time, or 60 seconds because of API calls
def sleeptime(calls,t=1):
    return 60 if not calls else t

# Append a message to the log file and save it
def update_log(message,filename):
    file = open(filename,"a")
    file.write(message+'\n')
    file.close()

# Returns n weights for mean gradients following some function
# Provide the power of the polynomial you want to fit for
def get_weights(f,n,k=0):
    x = np.linspace(k,1,n)
    y = f(x)
    y /= np.sum(y)
    return y

# Simple MinMax Scaler for values between 0 and 1
def minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn)/(mx-mn)

# Mean absolute percent error
def mape(y,y_hat):
    return np.mean(np.abs((y_hat - y)/y))

# Returns a list of the stocks meeting filtering criteria
# Pass how many you want to get total, and time willing to wait (in minutes)
def filter_stocks(api, av, n=100, t=10, low=1, high=10, exchange=None):

    # Stocks we already own
    currently_own = [p.symbol for p in api.list_positions()]

    # Initial filter for active, tradable and exchange
    tradable_symbols = [stock for stock in api.list_assets(status='active') if stock.tradable]
    if exchange is not None:
        tradable_symbols = [stock.symbol for stock in tradable_symbols if stock.exchange == exchange]
    else:
        tradable_symbols = [stock.symbol for stock in tradable_symbols]

    # Shuffle and begin to filter based on price range
    random.shuffle(tradable_symbols)
    symbols = []
    calls = 0
    for symbol in tradable_symbols:

        # Check if gotten the desired stocks or went above time
        if not n or not t:
            break

        # Check if we already have shares for this stock, if so, continue
        if symbol in currently_own:
            continue

        # Get current price and add one to calls (for getting price)
        try:
            calls = (calls + 1) % (APCA_MAX_CALLS - 1)  # Minus 1 because of initial call up top for safety
            close = api.get_barset(symbol, '1Min', 1)[symbol][0].c
        except Exception:
            if not calls: # Duplicate in case api fail
                time.sleep(60)
                t -= 1
            continue
        
        # Get close (set to Inf if had no price to ensure it's not added)
        if low <= close <= high:
            symbols.append(symbol)
            n -= 1

        # Check if have to sleep to not exceed max calls
        if not calls:
            time.sleep(60)
            t -= 1

    # Remember to sleep for a minute before returning
    time.sleep(60)
    return symbols


# Curve fitting
def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


# Function that returns the stocks recommended to buy
# Takes filtered symbols, and 2 weight vectors.
# w1: Weight vector for the mean gradients
# w2: Weight vector for the curve fits (should match length of polys)
# Could pass weights for volatility calculations, but I think constant is best
def recommend_to_buy(api, av, symbols, w1, w2, tw=0.7, vw= 0.3, N=30, M = 5, day_spacing=2,
                     polys=[linear, quadratic, cubic]):

    # Get the current date and create the directory for this date
    current_date = datetime.date.today()
    date_string = current_date.strftime('%Y-%m-%d')
    dirname = date_string + '_Recommend'
    if os.path.exists(dirname): # Delete and remake to clear
        os.rmdir(dirname)
    os.mkdir(dirname)

    # Get a date we want prices from
    # To ensure tight spacing between recent pricing
    days_go_back = int(N*1.5*day_spacing)
    delta_date = datetime.timedelta(days=days_go_back)
    lower_date = (current_date - delta_date).strftime('%Y-%m-%d')

    # Lists all things to track
    syms = []
    trend_strengths = []
    daily_volatilities = []
    intraday_volatilities = []
    all_closes = []
    all_intraday_closes = []

    # Begin to look up historical data for each up to date specified
    calls = 0
    tracking_number = 0
    track_after = 5
    for symbol in symbols:

        # Track
        tracking_number += 1
        if tracking_number % track_after == 0:
            print('Working on stock #{}'.format(tracking_number))

        # Grab raw values, filter by date, turn to np and reverse for proper order
        try:
            calls = (calls + 1) % AV_MAX_CALLS # Put before so it still increments if we exit try clause
            hist = av.historic_quotes(symbol)
            hist = np.array([list(hist.get(i).values()) for i in hist if i >= lower_date],
                   dtype=np.float)[::-1]
            hist[N] # See if we at least have (N+1) of history, so not dead stock
            closes = hist[-N:, CLOSE_INDEX] # If so, certainly we can get last N
        except Exception: # Duplicate sleep api call check in except, since we continue
            if not calls:
                time.sleep(60)
            continue

        # TODO: Change this layout here. Make it so after we have enough historical daily data,
        # and intraday data, we simply pass all this data to various tests.
        # Tests will be trend stengths, volatility tests, indicators ect.
        # If a stock passes all tests, we calculate it's score and add it to the list
        # Also, take this out of utils, put all recommendations stuff in another file

        # Passed the tests, normalize all close values
        closes /= np.sum(closes)

        # Begin to curve fit
        strengths = []
        xs = np.linspace(0, 1, N)
        for poly in polys:

            # Get coefs for this curve fit
            coefs, _ = opt.curve_fit(poly, xs, closes)

            # Grab all values given from the fit (using more values but same bounds)
            xs_dense = np.linspace(0, 1, len(w1))
            h = xs_dense[1] - xs_dense[0]
            y = poly(xs_dense, *coefs)

            # Add mean gradient to trend_strength
            strengths.append(np.sum(np.gradient(y, h) * w1))

        # Get the mean of the all the trend strengths
        trend_strength = np.sum(np.array(strengths) * w2)

        # ** If the trend strength if positive, continue onto intraday volatility
        if trend_strength > 0:

            # See if we can get intraday volatilities, if not then skip this stock
            if not calls: # Sleep if need to before another call
                time.sleep(60)
            try:
                calls = (calls + 1) % AV_MAX_CALLS
                intradays = av.intraday_quotes(symbol,'1min')
                intraday_closes = np.array([list(sample.values()) 
                              for sample in intradays.values()],dtype=np.float)[::-1,CLOSE_INDEX]
                intraday_closes[N] # See if we can get at least N
            except Exception:
                if not calls: # Check for sleep again because of continue
                    time.sleep(60)
                continue

            # Append everything so far, this stock is good
            syms.append(symbol)
            trend_strengths.append(trend_strength)

            # Daily voli calc
            daily_volatility = np.mean(np.abs(np.gradient(closes)))
            daily_volatilities.append(daily_volatility)

            # Finish intraday voli calc 
            intraday_gradient = np.gradient(intraday_closes)
            intraday_gradient_ups = intraday_gradient[intraday_gradient > 0]
            intraday_gradient_downs = intraday_gradient[intraday_gradient < 0]
            intraday_volatility = (np.mean(np.abs(intraday_gradient)),np.mean(intraday_gradient_ups),np.mean(intraday_gradient_downs))
            intraday_volatilities.append(intraday_volatility)

            # Add to closes and intraday_closes
            all_closes.append(closes)
            all_intraday_closes.append(intraday_closes)

        # Sleep if hit max calls
        if not calls:
            time.sleep(60)

    # Rescale the trend_strengths and daily_volatilities to be between 0 and 1
    trend_strengths = minmax(np.array(trend_strengths))
    daily_volatilities = minmax(np.array(daily_volatilities))

    # Take the average to figure out final score (use 1-dv, because 1 is bad and 0 is good
    # Use provided weights (that sum to 1 of course)
    scores = tw*trend_strengths + vw*(1-daily_volatilities)

    # Argsort on scores
    sorted_order = np.argsort(scores)[::-1]

    # Reshift all values by this argsort and grab the top M
    syms = np.array(syms)[sorted_order][:M]
    trend_strengths = trend_strengths[sorted_order][:M]
    daily_volatilities = daily_volatilities[sorted_order][:M]
    intraday_volatilities = np.array(intraday_volatilities)[sorted_order][:M] # np
    all_closes = np.array(all_closes)[sorted_order][:M] #np
    all_intraday_closes = np.array(all_intraday_closes)[sorted_order][:M] #np


    # Start plotting these values with the title as values 
    # Find mn and mx values for ylims
    ylow = np.min(all_closes)
    yhigh = np.max(all_closes)
    for i in range(len(syms)):

        # Create sub plots, one for daily one for intraday
        fig,axs = plt.subplots(3,figsize=(16,22))
        fig.suptitle(syms[i],fontsize=18)
        axs[0].plot(np.arange(len(all_closes[i])),all_closes[i])
        axs[0].set_title('DAILY -> TREND: {} VOL: {}'.format(np.around(trend_strengths[i],decimals=5),
                                                 np.around(daily_volatilities[i],decimals=5)))
        axs[0].set_ylim([ylow,yhigh])
        axs[1].set_title('DAILY: Zoom in look')
        axs[1].plot(np.arange(len(all_closes[i])),all_closes[i])
        axs[2].plot(np.arange(len(all_intraday_closes[i])),all_intraday_closes[i])
        axs[2].set_title('INTRADAY -> VOL: {} UPS: {} DOWNS: {}'.format(np.around(intraday_volatilities[i,0],decimals=5),
                                                                 np.around(intraday_volatilities[i,1],decimals=5),
                                                                 np.around(intraday_volatilities[i,2],decimals=5)))

        # Save the plot for each top stock (and close)
        plt.savefig(dirname+'/'+'Stock_'+str(i+1))
        plt.close()

    # Returns nothing, plots are enough
    time.sleep(60)
