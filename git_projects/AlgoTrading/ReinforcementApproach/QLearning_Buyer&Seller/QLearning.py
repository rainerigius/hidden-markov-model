from IMPORTS import *
from CLASSES import *
from UTILITY_FUNCS import *

# Set layers for BUY NN
buy_input_ = 6
buy_hidden1 = 256
buy_hidden2 = 128
buy_output_ = 2

# Build the buy model
class Buy_Model(nn.Module):
    def __init__(self):
        super(Buy_Model,self).__init__()
        self.input_ = nn.Linear(buy_input_,buy_hidden1)
        self.hidden1 = nn.Linear(buy_hidden1,buy_hidden2)
        self.hidden2 = nn.Linear(buy_hidden2,buy_output_)

    # Standardize the input (not the normalized time)
    def forward(self,x):
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
buyer = Buy_Model()
buy_lr = 0.0001
buy_optimizer = torch.optim.Adam(buyer.parameters(),lr=buy_lr)

# The loss function for buyer
buy_loss_fn = nn.MSELoss()

# Now create the seller
sell_input_ = 6
sell_hidden1 = 256
sell_hidden2 = 128
sell_output_ = 2

# Build the sell model
class Sell_Model(nn.Module):
    def __init__(self):
        super(Sell_Model,self).__init__()
        self.input_ = nn.Linear(sell_input_,sell_hidden1)
        self.hidden1 = nn.Linear(sell_hidden1,sell_hidden2)
        self.hidden2 = nn.Linear(sell_hidden2,sell_output_)

    # Standardize the input (not the normalized time)
    def forward(self,x):
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
seller = Sell_Model()
sell_lr = 0.0001
sell_optimizer = torch.optim.Adam(seller.parameters(),lr=sell_lr)

# The loss function for seller
sell_loss_fn = nn.MSELoss()

# Set up an epislon greedy stratedgy (start with very high epsilon, keep linear)
epsilon = 1
epsilon_min = 0.01
epochs = 2500
discount = 0.95
delta = (epsilon - epsilon_min)/(epochs-1)
delta = (epsilon_min/epsilon)**(1/epochs)
clips = epochs + 1

# Set up experience replay
buy_deque_len = 500
buy_batch_size = 50
buy_transitions = deque(maxlen=buy_deque_len)

sell_deque_len = 500
sell_batch_size = 50
sell_transitions = deque(maxlen=sell_deque_len)

# Track all things
time_lasted = []
epsilons = []
epchs = []
profits = []
times = []
prices = []
buy_times = []
buy_prices = []
sell_times = []
sell_prices = []

# Begin to start episodes (create env first)
env = Environment('GOOGL')
for epoch in range(1,epochs+1):


    # Set up
    state = env.reset()
    done = False

    # This is for tracking and plotting
    time = []
    price = []
    buy_time = []
    sell_time = []
    buy_price = []
    sell_price = []

    # While we haven't terminated yet
    while not done:

        # Print the time and epoch
        #print('Epoch:',epoch,'\tTime:',env.day.t)

        # Grab if has stock (need original value after we take a step)
        has_stock = env.has_stock

        # With no grad, grab action values (check what network to use)
        with torch.no_grad():
            if has_stock:
                action_values = seller(torch.from_numpy(state).float())
            else:
                action_values = buyer(torch.from_numpy(state).float())

        # Use epsilon greedy strategy
        # TODO: Change it to NOT be able to select highest
        # TODO: Add has_stock to sell (sell has to be 2)
        if np.random.rand() < epsilon:
            action = env.sample(has_stock)
        else:
            action = torch.argmax(action_values) + has_stock

        # Add this time and price
        time.append(env.day.t)
        price.append(env.day.open)

        # Add these actions
        if action == BUY:
            buy_time.append(env.day.t)
            buy_price.append(env.day.open)
        elif action == SELL:
            sell_time.append(env.day.t)
            sell_price.append(env.day.open)

        # Take a step with this action and get observables
        state_prime,reward,done = env.step(action)

        # Add everything to transition
        # TODO: Add done, so we can not zero out the expected reward for terminal state
        # TODO: Possibly minus 1 from sell action
        # TODO: Add has stock
        if has_stock:
            sell_transitions.append((state,action-int(has_stock),reward,state_prime,has_stock,done))
        else:
            buy_transitions.append((state,action-int(has_stock),reward,state_prime,has_stock,done))

        # If we can train, do so
        if len(buy_transitions) > buy_batch_size:
            train_model(buyer,buy_transitions,buy_batch_size,buy_optimizer,
                        buy_loss_fn,seller,discount)
        if len(sell_transitions) > sell_batch_size:
            train_model(seller,sell_transitions,sell_batch_size,sell_optimizer,
                        sell_loss_fn,buyer,discount)

        # Set state as state_prime
        state = state_prime

    # Track
    print('Epoch: {} with epsilon {} lasted {} minutes'.format(epoch,epsilon,env.day.t - TRADE_OPEN))
    time_lasted.append(env.day.t - TRADE_OPEN)

    # Day (episode) has ended, need to sell stock if we still own it (env.shares will be 0 if we don't own any so it works)
    env.equity += env.shares*env.day.open # Just use open, whatever

    # Decrease epsilon (but first save the one used for the episode)
    eps_used = epsilon
    if np.random.rand() < epsilon:
        epsilon -= (epsilon - epsilon_min)/(epochs - epoch)
    else:
        epsilon *= (epsilon_min/epsilon)**(1/(epochs-epoch+1))

    # Show a plot if applicable
    if epoch % clips == 0:

        epsilons.append(eps_used)
        epchs.append(epoch)
        profits.append(env.equity - 20000)
        times.append(time)
        prices.append(price)
        buy_times.append(buy_time)
        buy_prices.append(buy_price)
        sell_times.append(sell_time)
        sell_prices.append(sell_price)

for i in range(len(times)):
    plt.plot(times[i],prices[i])
    plt.scatter(buy_times[i],buy_prices[i])
    plt.scatter(sell_times[i],sell_prices[i])
    plt.title('Epoch: {} with epsilon {} had {} profit lasting {} minutes'.format(epchs[i],np.around(epsilons[i],decimals=3),
              np.around(profits[i],decimals=3),times[i][-1] - TRADE_OPEN))
    plt.legend(('Prices','Buy','Sell'))
    plt.show()

plt.scatter(range(len(time_lasted)),time_lasted)
plt.show()

torch.save(buyer.state_dict(),'buyer.pt')
torch.save(seller.state_dict(),'seller.pt')
