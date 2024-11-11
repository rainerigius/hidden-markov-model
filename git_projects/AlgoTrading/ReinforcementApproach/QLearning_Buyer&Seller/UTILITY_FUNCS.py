from IMPORTS import *
from CLASSES import *

# Grabs a realistic value using all prices of the stock for the day
def action_price(intraday,idx,std_percent=0.1):
    price_points = intraday[idx,OPEN_INDEX:VOLUME_INDEX]
    weighted_mean = price_points*PRICE_WEIGHTS
    return np.random.normal(weighted_mean,weighted_mean*std_percent)

# Computes mean gradient
def mean_gradient(data):
    return np.mean(data[1:] - data[0:-1])

# Gets the normalized time, between 0 and 1 (with respect to trading open and open)
def normalized_time(current_time):
    return (current_time - TRADE_OPEN)/(TRADE_CLOSE - TRADE_OPEN)

# Standardize data (comes in as Tensor which as funcs built in)
def standardize(data):
    return (data - data.mean())/data.std()

# Returns only items within market time
def within_market_times(data):

    # Need bools for times >= 9:30 and <= 4
    open_bools = np.array(data.apply(lambda x: x.name.time().hour*60 + x.name.time().minute >= MARKET_OPEN,axis=1))
    close_bools = np.array(data.apply(lambda x: x.name.time().hour*60 + x.name.time().minute <= MARKET_CLOSE,axis=1))
    bools = np.logical_and(open_bools,close_bools)
    return data[bools]

# Returns two lists.
# 1) Contains all intraday data for every day (minus the first, in numpy)
# 2) The open value for 9:29 on that day. (Might come from day before, that's
       # why we drop the first day just in case)
# Just pass the symbol, to do multiple stocks
def get_days(symbol):

    # Get the Alpha Vantage data for the historical data
    api = tradeapi.REST(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'v2')
    av = api.alpha_vantage

    # Get all the intraday data
    historical_data = av.intraday_quotes(symbol,'1min','full','pandas')

    # Create two empty lists to return
    intradays = []
    opens = []

    # Get a list of unique days
    unique_days = np.unique(list(map(lambda x: x.date(), historical_data.index)))

    # Start off with getting previous day
    bools = np.array(historical_data.apply(lambda x: x.name.date() == unique_days[0], axis=1))
    previous_intraday = historical_data[bools][::-1] # Remember to reverse!

    # Begin to iterate rest of the days (skip first day)
    for day in unique_days[1::]:

        # Get current intraday data
        bools = np.array(historical_data.apply(lambda x: x.name.date() == day, axis=1))
        current_intraday = historical_data[bools][::-1] # Rememebr to reverse!

        # Now we need to find the last closing value before 9:30 for the current day.
        # Which can either be the last open of the previous day, or the most recent closing
        # value of the current day, before 9:30
        previous_day_close = previous_intraday.iloc[-1][CLOSE]
        bools = np.array(current_intraday.apply(lambda x: x.name.time().hour * 60 + x.name.time().minute < MARKET_OPEN, axis=1))
        current_intraday_premarket = current_intraday[bools]
        open = previous_day_close if current_intraday_premarket.size == 0 else current_intraday_premarket.iloc[-1][OPEN]

        # Trim current_intraday to within market, add time and convert to numpy
        intraday = within_market_times(current_intraday)
        intraday.insert(0,'0. time',np.array(list(intraday.apply(lambda x: x.name.time().hour*60 + x.name.time().minute,axis=1))))
        intradays.append(np.array(intraday)) # Convert to numpy!
        opens.append(open)

        # Set previous to current
        previous_intraday = current_intraday

    # Return both the lists, intradays and opens (last open recorded before 9:30)
    return intradays,opens

# Function to get the open history
def get_open_history(data,open,t):

    # Go through, and grab the last values until TRADE_OPEN, interpolate as needed
    retval = deque(maxlen=NUM_OPEN_HISTORY)
    idx = 0
    while t < TRADE_OPEN:

        # Get the time of the next piece of data
        next_t = data[idx,TIME_INDEX]

        # If this time is not 1 minute later from current time,
        # must clone values, and not iterate any values (idx, open ect) except t (by 1)
        if next_t - t > 1:
            retval.append(data[idx,CLOSE_INDEX])

        # Otherwise, get the new open and increase idx
        else:
            retval.append(open)
            open = data[idx,OPEN_INDEX]
            idx += 1

        # Increment t by 1 no matter what
        t += 1

    # Return the open history
    return retval

# Get current moving averages
def get_moving_averages(data,open,t,offset):

    # Use a dictionary
    retval = dict()
    for k in MOVING_AVG_KS:
        retval[k] = deque(maxlen=k)

    # Set up and begin to calculate
    idx = 0
    while t < TRADE_OPEN-offset:

        # Get next time
        next_t = data[idx,TIME_INDEX]

        # If next actual time isn't 1 minute..
        # Use same value, don't increment anything
        if next_t - t > 1:
            for k in MOVING_AVG_KS:
                retval[k].append(data[idx,CLOSE_INDEX])

        # Otherwise get new value
        else:
            for k in MOVING_AVG_KS:
                retval[k].append(open)
            open = data[idx,OPEN_INDEX]
            idx += 1

        # t no matter what
        t += 1

    # Return moving averages
    for k in MOVING_AVG_KS:
        retval[k] = np.mean(retval[k])
    return retval

# Returns index to start the trade day on
def get_start_index(data,t):

    # idx, go while time <= trade open
    idx = 0
    while t < TRADE_OPEN:
        next_t = data[idx,TIME_INDEX]
        if next_t - t == 1:
            idx += 1
        t += 1

    return idx

# Trains the model
# TODO: Right now we literally hardcoded the action space as 2
def train_model(model,transitions,batch_size,optimizer,
                loss_fn,other_model,discount):

    # Grab the batch
    batch = random.sample(transitions,batch_size)

    # Extract all the parts
    # TODO: Remember, actions are both 0 or 1.
    states = torch.Tensor([s for (s,_,_,_,_,_) in batch])
    actions = [a for (_,a,_,_,_,_) in batch]
    rewards = torch.Tensor([r for (_,_,r,_,_,_) in batch])
    state_primes = torch.Tensor([sp for (_,_,_,sp,_,_) in batch])
    has_stocks = torch.Tensor([hs for (_,_,_,_,hs,_) in batch])
    dones = torch.Tensor([d for (_,_,_,_,_,d) in batch])

    # Normalize states and stateprimes (normalized time isn't affected)
    #d = states.shape[0]
    #states = torch.cat((states[:,0].reshape(d,-1),standardize(states[:,1:])),dim=1)
    #dp = state_primes.shape[0]
    #state_primes = torch.cat((state_primes[:,0].reshape(d,-1),standardize(state_primes[:,1:])),dim=1)

    # Get the targets first (before actions is changed) no grad
    with torch.no_grad():
        action_values_prime = torch.empty((batch_size,2))
        for i in range(batch_size):
            if actions[i] == has_stocks[i]:
                action_values_prime[i] = other_model(state_primes[i])
            else:
                action_values_prime[i] = model(state_primes[i])
        targets = rewards + (torch.max(action_values_prime,dim=1)[0]*discount)*(1-dones)

    # Get action value for all states (with grad)
    action_values = model(states).view(-1)

    # Scale up actions to get by number to get index for flattened states
    actions = [i*2 + actions[i] for i in range(len(actions))]

    # Grab these predictions
    predictions = action_values[actions]

    # Compute the loss and backprop ect (detach for safety)
    loss = loss_fn(predictions,targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
