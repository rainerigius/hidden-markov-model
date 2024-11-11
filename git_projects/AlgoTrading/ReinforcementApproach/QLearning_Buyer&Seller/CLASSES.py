from IMPORTS import *
from UTILITY_FUNCS import *

# The day class, will handle all interaction with the data.
# To be used heavily by the Environment Class (instance variable for it)
class Day:

    # NOTE, after open value and time are used to get histories,
    # Must set them to values for 10AM. and increment idx to proper index
    def __init__(self,intraday,open):

        # Set the current time as 9:29
        self.t = MARKET_OPEN - 1

        # All the intraday data, and the open value at current time (9:29)
        self.intraday = intraday
        self.open = open

        # Set the last index of data
        # NOTE: May not use it (trade open is before it)
        #       Or, may extend past it (it happens before trade open)
        self.last_index = len(self.intraday)

        # Get the open_history (last 3 values before previous)
        # Use a deque for this, easiest way
        # NOTE (it's just the values at time (9:57, 9:58, 9:59) if NUM_OPEN_HISTORY is 3
        self.open_history = get_open_history(self.intraday,self.open,self.t)

        # Get the current moving averages (up to 9:59)
        self.moving_averages = get_moving_averages(self.intraday,self.open,self.t,0)

        # Get moving average histories (abuse calls to moving averages)
        # Its stupid and slow but easy to understand so whatever
        self.moving_averages_histories = dict()
        for k in MOVING_AVG_KS:
            self.moving_averages_histories[k] = deque(maxlen=NUM_MOVING_AVG_HISTORY)
        for i in range(NUM_MOVING_AVG_HISTORY,0,-1):
            avgs = get_moving_averages(self.intraday,self.open,self.t,i)
            for k in MOVING_AVG_KS:
                self.moving_averages_histories[k].append(avgs[k])

        # Get the current index to start on
        # Which will be the last index with a time <= 10
        self.idx = get_start_index(self.intraday,self.t)

        # Finally set the open, and time accordingly (-1 for actual open value <= 10)
        self.open = self.intraday[self.idx-1,OPEN_INDEX]
        self.t = TRADE_OPEN

# The environment class
class Environment:

    # Init to get all intraday data
    def __init__(self,symbol):

        # The intraday and open data
        self.intradays,self.opens = get_days('GOOGL')

        # day is set to None until reset
        self.day  = None

        # Size of intradays/opens to generate random number
        self.day_span = len(self.intradays)

        # The action space and action space size
        self.action_space = np.arange(3)
        self.action_space_n = self.action_space.size

        # Keeps track of the shares and buying prices
        self.shares = 0
        self.buying_price = None
        self.has_stock = False

        # Sets initial equity
        self.equity = 20000

    # The reset function, returns the state
    def reset(self,day_index=None):

        # Reset for fresh day
        self.shares = 0
        self.buying_price = None
        self.has_stock = False
        self.equity = 20000

        # Grab day
        if day_index == None:
            day_index = np.random.randint(self.day_span)

        # Set the current day
        self.day = Day(self.intradays[day_index],self.opens[day_index])

        # Get observation space
        open_gradient = np.array([mean_gradient(np.array(self.day.open_history))])
        moving_averages = np.array(list(self.day.moving_averages.values()))
        moving_averages_gradient = np.apply_along_axis(mean_gradient,1,np.array(list(self.day.moving_averages_histories.values())))
        open_vs_ma = (self.day.open - moving_averages)/moving_averages
        ma_vs_ma = np.array([(moving_averages[1] - moving_averages[0])/moving_averages[0]])

        # Return it all
        return np.concatenate((open_vs_ma,open_gradient,ma_vs_ma,moving_averages_gradient))

    # The step function, takes the action and returns a new state,reward,and done
    # TODO: Add in getting punished if action isn't sell while holding stocks at last time
    def step(self,action):

        # Perform the action in the current state
        # Get the close price for the day (when the trade will actually go through)
        # This is the previous day, because remember idx is on the next set of data
        close = self.day.intraday[self.day.idx-1,CLOSE_INDEX]
        if action == BUY:
            self.shares = divmod(self.equity*BUY_PERCENTAGE,self.day.open)[0]
            self.buying_price = close
            self.equity -= self.shares*self.buying_price
            self.has_stock = True
            reward = 0
            reward = reward if self.day.t < TRADE_CLOSE else -10
        elif action == SELL:
            net = self.shares*close
            cost = self.shares*self.buying_price
            self.equity += net
            self.shares = 0
            self.buying_price = None
            self.has_stock = False
            reward = (net - cost)/cost*100
            reward = 1 if reward >= 0 else -1
            time_scaler = 1 - normalized_time(self.day.t) + 1e-10
            reward = reward*time_scaler
        elif action == HOLD:
            if self.has_stock:
                net = self.shares*close
                cost = self.shares*self.buying_price
                reward = 0
            else:
                reward = 0
            reward = reward if self.day.t < TRADE_CLOSE or not self.has_stock else -10

        # Now update the all variables
        # But first, get the new time and idx
        next_t = self.day.intraday[self.day.idx,TIME_INDEX]
        has_next_day = next_t - self.day.t == 1
        self.day.t += 1

        # Do moving averages
        for k in MOVING_AVG_KS:
            self.day.moving_averages_histories[k].append(self.day.moving_averages[k])
            self.day.moving_averages[k] += (self.day.open - self.day.open_history[-1])/k

        # Do opens
        self.day.open_history.append(self.day.open)
        self.day.open = self.day.intraday[self.day.idx,OPEN_INDEX] if has_next_day else self.day.intraday[self.day.idx,CLOSE_INDEX]

        # Update idx
        self.day.idx += has_next_day

        #Package everything
        open_gradient = np.array([mean_gradient(np.array(self.day.open_history))])
        moving_averages = np.array(list(self.day.moving_averages.values()))
        moving_averages_gradient = np.apply_along_axis(mean_gradient,1,np.array(list(self.day.moving_averages_histories.values())))
        open_vs_ma = (self.day.open - moving_averages)/moving_averages
        ma_vs_ma = np.array([(moving_averages[1] - moving_averages[0])/moving_averages[0]])
        observations = np.concatenate((open_vs_ma,open_gradient,ma_vs_ma,moving_averages_gradient))

        # Account for end of data, has next will be false because the difference between t and getting time will be 0
        end_of_data = self.day.idx == self.day.last_index
        self.day.idx -= end_of_data
        done = self.day.t > TRADE_CLOSE or reward < 0
        return observations,reward,done

    # Needs to know if buying or selling
    def sample(self,is_selling):
        return np.random.randint(2) + is_selling
