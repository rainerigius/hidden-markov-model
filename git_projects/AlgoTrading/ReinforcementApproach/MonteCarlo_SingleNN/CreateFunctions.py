import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import torch
import random
from collections import deque
from ENV_Variables import *

# Takes a timestamp object, and returns if the market is closed
def market_closed(timestamp):
    t = timestamp.time()
    t_converted = t.hour*60 + t.minute
    if t_converted < OPEN_TIME or t_converted > CLOSE_TIME:
        return True
    return False

# Takes returns only timestamps that are within the market hours
def trim_timestamps(timestamps):
    bools = list(map(lambda timestamp: market_closed(timestamp),timestamps))
    return timestamps[bools]

# Returns the data only within market open
def trim_to_market_open(data):
    return data.drop(index=trim_timestamps(data.index))

# Fills in missing data with no changes
def fill_no_changes(data):
    # Start with empty data and relevant info we need to track
    retval = pd.DataFrame()
    timestamps = data.index
    index = 0

    # While we're still iterating this data (only go til second last)
    # Can't interpolate with nothing else left
    while index < len(data)-1:
        # Get the current day (integer) and converted time, and the data for this index
        day = timestamps[index].date().day
        converted_time = timestamps[index].time().hour * 60 + timestamps[index].time().minute
        series = data.iloc[index]

        # Append this information to the dataframe
        retval = retval.append(series)

        # Get the next day and data in dataframe
        next_day = timestamps[index+1].date().day
        next_time = timestamps[index + 1].time().hour * 60 + timestamps[index + 1].time().minute

        # If it's the actual next day..
        if next_day != day:

            # Clone and append the values for the last trade til the market closes
            minutes_to_add = 1
            while converted_time < CLOSE_TIME:
                next_data = pd.Series(data=np.array(series),index=data.columns,name=timestamps[index]+pd.Timedelta(minutes=minutes_to_add),dtype=series.dtype)
                retval = retval.append(next_data)
                converted_time += 1
                minutes_to_add += 1

            # Clone and append the last trade from next day market open, until the next time
            # Don't have to worry about entire days going by with no trading
            # Nothing happens!
            converted_time = OPEN_TIME # Reset from use, already have next_time
            open_timestamp = timestamps[index+1]
            open_timestamp = open_timestamp.replace(hour=9, minute=30, second=0)
            minutes_to_add = 0
            while converted_time < next_time:
                next_data = pd.Series(np.array(series),index=data.columns,name=open_timestamp + pd.Timedelta(minutes=minutes_to_add),dtype=series.dtype)
                retval = retval.append(next_data)
                minutes_to_add += 1
                converted_time += 1

        # It's the same day, but check if it's not subsequent time (respect to minutes)
        elif next_time - converted_time > 1:

            # Append for the amount of times needed
            minutes_to_add = 1
            for i in range(next_time - converted_time - 1):
                next_data = pd.Series(np.array(series),index=data.columns,name=timestamps[index] + pd.Timedelta(minutes=minutes_to_add),dtype=series.dtype)
                retval = retval.append(next_data)
                minutes_to_add += 1

        # No matter what we do, increment index and continue
        index += 1

    # Return the new dataframe
    return retval

# Wrapper function to do all the data cleaning
def clean_data(data):
    return fill_no_changes(trim_to_market_open(data[::-1]))

# Computes the moving average for all values for various increments.
# Returns a dictionary for all moving averages (all k's)
def moving_averages(data,ks,history_len):

    # Create empty dictionary to append to
    moving_averages = dict()

    # For every k, compute and add
    for k in ks:
        averages = deque(maxlen=history_len)
        for i in range(history_len-1,-1,-1):
            averages.append(np.mean(data.iloc[-1-i:-1-k-i:-1]['4. close'], axis=0))
        moving_averages[k] = averages

    # Return moving_averages, last value array and new data
    max_k = np.max(ks)
    return moving_averages

# Get the normalized time
def normalized_time(time):
    return (time - OPEN_TIME) / (CLOSE_TIME - OPEN_TIME)

# Function for training on a batch for MC
# Accepts transitions (s,a,r) batchsize, and all relevant things for model
def train_model(transitions,batch_size,model,optimizer,loss_fn,num_actions):

    # Grab a batch from transitions
    batch = random.sample(transitions,batch_size)

    # Get all the the parts from the transition
    states = torch.Tensor([s for (s,a,r) in batch])
    actions = [a for (s,a,r) in batch]
    rewards = [r for (s,a,r) in batch]

    # Calculate returns
    returns = torch.Tensor(get_returns(rewards))
    #returns = (returns - returns.mean())/returns.std()

    # Feed all states to get action values and flatten
    action_values = model(states).view(-1)

    # Scale up actions by number to get index for flattened states
    actions = [i*num_actions + actions[i] for i in range(len(actions))]

    # Grab all action value predictions for the current state
    predictions = action_values[actions]

    # Compute the loss (the returns are the targets)
    loss = loss_fn(predictions,returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Gets the returns from the rewards
# REMEMBER TO REVERSE
def get_returns(rewards):

    R = 0
    returns = []
    for i in range(len(rewards)-1,-1,-1):

        R = R*0.99 + rewards[i]
        returns.append(R)

    return returns[::-1]
