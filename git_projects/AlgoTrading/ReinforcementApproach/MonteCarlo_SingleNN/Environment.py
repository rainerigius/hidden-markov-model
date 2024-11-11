import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from ENV_Variables import *
import pickle
import random
from collections import deque
from CreateFunctions import *

# The Environment class. Contains the standard openAIGym functions
class Environment:

    # Holds the api and av object, intraday data, starting moving averages,
    # and current index for data. Also what symbol of course!
    def __init__(self,key_id,secret_key,base_url,symbol,ks=[10,50,100,200],
                 bp=[0.5],equity=20000.0,history_len=4,filename=''):

        # The api and av object, will be used by everything else in init
        self.api = tradeapi.REST(key_id,secret_key,base_url,'v2')
        self.av = self.api.alpha_vantage

        # Equity started with
        self.equity = equity
        self.starting_equity = equity

        # Two arrays, one for number of shares bought,
        # the other for price bought at
        self.shares_owned = np.array([])
        self.buying_price = np.array([])

        # Set the ks (for moving averages), (sorted)
        self.ks = np.sort(ks)
        self.idx = np.max(self.ks) + history_len - 1

        # Set the buy percentages as np.array
        # And get how many for max buy action integers
        # As well as action_space and action_space size
        self.bp = np.array(bp)
        self.num_buying_actions = self.bp.size
        self.action_space = np.arange(self.num_buying_actions+2)
        self.num_actions = self.action_space.size

        # All the intraday data available for this symbol as pd (every minute)
        # Flips data (earlier -> latest), and cleans it.
        # Load through pickle if filename not none
        if len(filename) > 0:
            self.data = pd.read_pickle(filename)
        else:
            self.data = clean_data(self.av.intraday_quotes(symbol, '1min', 'full', 'pandas'))

        # Testing only first day
        bools = list(map(lambda x: x.date().day == 14, self.data.index))
        self.data = self.data.iloc[bools]

        # Grab length of data
        self.data_len = len(self.data)

        # Get all the moving average data for every field.
        self.moving_averages = moving_averages(self.data.iloc[0:self.idx], self.ks, history_len)

        # Get last values used for moving averages
        self.last_values = np.array(self.data.iloc[self.idx-2]['4. close'])

        # Get the close value
        self.close = np.array(self.data.iloc[self.idx-1])[3]

        # Keeps a history of the last close values (including current)
        self.close_history = deque(maxlen=history_len)
        self.close_history.extend(np.array(self.data.iloc[self.idx-history_len:self.idx])[3])

        # Get current time
        timestamp = self.data.iloc[self.idx-1].name
        self.current_time = timestamp.time().hour*60 + timestamp.time().minute

    # Returns the current state as an np.array
    # Updates index to next
    def next_state(self):

        # Wrap equity in np.array
        equity_wrap = np.array([self.equity])

        # Next datum of data
        datum = self.data.iloc[self.idx]

        # Timestamp for time and grab the time normalized for max time
        timestamp = datum.name
        self.current_time = timestamp.time().hour*60 + timestamp.time().minute
        t = np.array([normalized_time(self.current_time)])

        # Convert datum to np.array (since we're done with it)
        datum = np.array(datum)

        # Calculates the total unrealized profit or loss (also set's the current close)
        # And close history
        # And last value
        self.last_values = self.close
        self.close = datum[3]
        self.close_history.append(self.close)
        profit_loss = np.sum((self.close - self.buying_price)*self.shares_owned)
        profit_loss = np.array([profit_loss])

        # Owns stock or not
        owns_stock = int(self.buying_price.size > 0 and self.shares_owned.size > 0)

        # Update moving averages before grabbing values for moving averages
        for key in self.moving_averages.keys():
            current = self.moving_averages[key][-1] + (self.close - self.last_values)/key
            self.moving_averages[key].append(current)

        # Set moving averages
        moving_average_values = np.array(list(self.moving_averages.values())).flatten()

        # Join everything but time together, then normalize
        retval = np.concatenate((np.array(self.close_history), moving_average_values)).astype(np.float64)

        # update idx (since were done, and return)
        self.idx += 1
        return retval

    # The reset function, simply calls next_state
    def reset(self):
        return self.next_state()

    # ************THREE ACTION FUNCTIONS TO BE USED IN STEP***********
    def buy_stocks(self,action):

        # Buy the stocks at closing price
        self.shares_owned = np.append(self.shares_owned,divmod(self.bp[action]*self.equity,self.close)[0])
        self.buying_price = np.append(self.buying_price,self.close)

        # Update equity
        self.equity -= self.shares_owned[-1]*self.buying_price[-1]

        # Return the reward for buying
        # Testing out, buying is an investment,
        # To enforce how much you need to make with the sell to get a positive reward,
        # make buying cost the inverse of that.
        # Below positively reinforces buys, when the selling price was 1% or more.
        # Add in the time, alleviate the pressure to make big returns
        # early, but keep them for late.
        return -0.01

    def sell_stocks(self):

        # Calculate total amount stocks sold for
        selling_total = np.sum(self.close*self.shares_owned)

        # Calculate buying_total
        buying_total = np.sum(self.buying_price*self.shares_owned)

        # Clear the arrays add to equity and return
        self.buying_price = np.array([])
        self.shares_owned = np.array([])
        self.equity += selling_total # Add what we actually got to equity

        # PL percentage
        return (selling_total - buying_total)/buying_total

    def hold(self):

        # To avoid buying and selling like crazy, holding will also have a reward.
        # Eh, keep 0 for now.
        return 0.0

    # The step functions, takes an action and returns new info
    def step(self,action):

        # Check if buying action
        if action < self.num_buying_actions:

            # Buy price as percentage
            # Function will update shares and buying price arrays and equity
            reward = self.buy_stocks(action)

        # Check if selling (set to num_buying_actions)
        elif action == self.num_buying_actions:

            # Sell all the stock
            # Frees both arrays (hold no stocks anymore), updates equity
            reward = self.sell_stocks()

        # Else we must do nothing, so do nothing (just call function for clarity)
        else:

            # Dummy call
            reward = self.hold()

        # Now we need to update and return the following
        # next_state,reward,sold,end_of_data
        next_state = self.next_state()
        sold = action == self.num_buying_actions
        end_of_data = self.idx == self.data_len

        return next_state, reward, sold, end_of_data

    # Returns the list of available actions (returns boolean array)
    # To be used with sample, and also picking highest Q value avaiable
    def available_actions_bools(self):

        # Can only buy if have nothing
        if self.buying_price.size == 0 and self.shares_owned.size == 0:
            # Checks if you can buy at least one stock for every percentage
            can_buy = np.array(list(map(lambda x: x > 0, divmod(self.equity*self.bp, self.close)[0])))
        else:
            can_buy = np.full(self.num_buying_actions,False)

        # Checks if you have stocks to sell
        can_sell = np.array([self.buying_price.size > 0 and self.shares_owned.size > 0])

        # You can always hold, so
        can_hold = np.array([True])

        # Concat together and return boolean array
        return np.concatenate((can_buy, can_sell, can_hold))

    # Sample the actions (possible to exclude highest value from sample for Q-learning)
    def sample(self, values=None):

        # Retrieve the actual possible actions
        bools = self.available_actions_bools()
        possible_actions = self.action_space[bools]

        # Trim out highest value if values not None
        if values is not None:
            values = values[bools]
            possible_actions = np.delete(possible_actions,values.argmax())

        # Choose one randomly if no actions are available
        # If possible_actions is empty because hold was the only available action,
        # and was dropped for being (by default) highest value, then still return hold
        return random.choice(possible_actions) if possible_actions.size > 0 else self.action_space[-1]


# Create a test environement
#env = Environment(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'GOOGL',
                  #filename='intraday.pkl',equity=20000)
