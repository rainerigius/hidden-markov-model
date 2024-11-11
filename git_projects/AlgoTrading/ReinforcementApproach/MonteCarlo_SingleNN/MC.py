# Make all necessary imports (importing Historical handles all np and pd ect)
from Environment import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

total_pl = []
eps = 1
diff = 0.01
while eps >= diff:
    # Create the environment first, so we have info to build NN
    env = Environment(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'GOOGL',
                      filename='intraday.pkl',equity=20000)

    # Create layer lengths for NN
    input_ = 20
    hidden1 = 140
    hidden2 =  60
    output_ = env.num_actions

    # Build the model
    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            self.input_ = nn.Linear(input_,hidden1)
            self.hidden1 = nn.Linear(hidden1,hidden2)
            self.hidden2 = nn.Linear(hidden2,output_)

        def forward(self,x):
            x = (x - x.mean())/x.std()
            x = F.leaky_relu(self.input_(x))
            x = F.leaky_relu(self.hidden1(x))
            x = self.hidden2(x)
            return x

    # Build model and set up optimizer
    model = Model()
    if eps < 1:
        model.load_state_dict(torch.load('themodel.pt'))
    lr = 0.0009
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # Set the loss function
    loss_fn = nn.MSELoss()

    # Set up variables for epsilon greedy strategy
    epsilon = eps
    epsilon_min = epsilon - diff + diff/100
    intervals = env.data_len - env.idx
    delta = (epsilon_min/epsilon)**(1/intervals)

    # Set variables for experience replay
    batch_size = 32
    deque_len = 370 # The entire day
    transitions = deque(maxlen=deque_len)

    # Set variable for stopping if data is used up
    end_of_data = False

    # Track the number of updates
    times_trained = 0

    # Daily profit loss and max equity
    daily_pl = []

    # Number of times bought and sell
    num_sold = 0
    num_bought = 0
    num_hold = 0

    # Time track of when we buy and sell
    buy_times = []
    buy_price = []
    sell_times = []
    sell_price = []

    print('Eps is:',eps)

    # While we still have data left (contiue to run episodes)
    while not end_of_data:

        # Set up the episode
        state = env.reset()
        sold = False
        episode_transitions = []

        # Grab end of data fresh for safety
        end_of_data = env.idx == env.data_len

        # While we have not sold or run out of data (marking end of episode)
        # NEED TO ADD AND TRACK END OF DAY HERE
        # If end of day (set as 15 minutes before market closes)
        # We still let the ELSE take care of it for not selling
        while not sold and not end_of_data and env.current_time < CLOSE_TIME:

            # Grab the action values for this state (no gradient)
            with torch.no_grad():
                action_values = model(torch.from_numpy(state).float())

            print(action_values)

            # Enforce epsilon greedy strategy
            if np.random.rand() < epsilon:
                action = env.sample(action_values)
            else:
                # Grab the available actions we have, then find max
                bools = env.available_actions_bools()
                action = env.action_space[bools][action_values[bools].argmax()]

            # Take a step with this action
            if action < env.num_buying_actions:
                num_bought += 1
                buy_times.append(env.current_time)
                buy_price.append(env.close)
            elif action == env.num_buying_actions:
                num_sold += 1
                sell_times.append(env.current_time)
                sell_price.append(env.close)
            else:
                num_hold += 1
            state_prime,reward,sold,end_of_data = env.step(action)

            # Update episode_transitions
            episode_transitions.append((state,action,reward))

            # If we have enough transitions, update
            if len(transitions) >= batch_size:
                train_model(transitions,batch_size,model,optimizer,
                            loss_fn,env.num_actions)

            # Decrease epsilon
            epsilon *= delta

            # Change state to stateprime
            state = state_prime

            # Printing for debuggin
            #print(env.equity)

        # The episode has ended
        # If it has ended because of a sell, calculate returns and add to transitions
        if sold:

            # Then append to transitions
            transitions.extend(episode_transitions)

        # Otherwise, it ended cause of the last data so see if we can sell
        # And then run on update just on this episode (potentially with special penalty for
        # not selling)
        else:

            # Check if we can sell
            if env.buying_price.size > 0 and env.shares_owned.size > 0:
                num_sold += 1
                sell_times.append(env.current_time)
                sell_price.append(env.close)

                # Force sell (observe reward)
                reward = env.sell_stocks()

                # Put into episode_trans
                episode_transitions.append((state,env.num_buying_actions,reward))

                # Add to transition
                transitions.extend(episode_transitions)

            # If can't sell, just do nothing.

        # If ended cause of current time or end_of_data then store the daily profit loss
        # And reset equity
        if env.current_time >= CLOSE_TIME or end_of_data:
            daily_pl.append(env.equity - 20000)
            env.equity = 20000

    # After everything is done, print total equity
    total_pl.append(np.sum(daily_pl))

    # Save the model
    torch.save(model.state_dict(),'themodel.pt')

    # Reduce eps
    eps -= diff

    # Print results
    print('Buy:',num_bought,'\tSell:',num_sold,'\tHold:',num_hold)
    print(daily_pl)

    # Make graphs
    if eps < diff*3:
        times = np.arange(OPEN_TIME,CLOSE_TIME)
        closing_prices = np.array(env.data['4. close'])

        plt.plot(times,closing_prices)
        plt.scatter(np.array(buy_times),np.array(buy_price))
        plt.scatter(np.array(sell_times),np.array(sell_price))
        plt.legend(('Close','Buy','Sell'))
        plt.show()


for params in model.parameters():
    print(params)

print('The bias is:',model.hidden2.bias)
x = np.arange(len(total_pl))
y = np.array(total_pl)
print(x)
print(y)
plt.scatter(x,y)
plt.show()
