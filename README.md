# DeepQHoldem
CS238/AA228 Final Project: Applying Deep Q-Learning to No-Limit Texas Hold’em Poker

DeepQHoldem is a Deep Q-Learning implementation for a poker playing agent. The agent is trained to play the game through a series of episodes, and the project provides functionality for generating data, training the agent, and evaluating its performance. Here is a demo showing the training pipeline and what it's like to play against our agent.

![deepq](https://github.com/jfrausto7/DeepQHoldem/assets/53204698/ff1c0c92-1d43-4b6e-b774-ac5aeff0a343)

## Prerequisites

- Python 3.x
- pip (Python package installer)
- Pytorch
- rlcard @ git+https://github.com/praneetbhoj/rlcard@master

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/jfrausto7/DeepQHoldem.git

2. Navigate to repository

## Usage

To train and evaluate on the agent:
  ```bash
  python main.py --generate
```


Optional arguments:

- --episodes: Number of episodes for training (default: 100000).
- --convergence_interval: Interval for calculating and printing the convergence rate during training (default: 1000).
- --players: Number of players in the game (default: 2).
- --chips: Number of starting chips for each agent/player (default: 1000).
- --dropout: Dropout rate for the DeepQAgent ANN (default: 0.2).
- --learning-rate: Learning rate for the DeepQAgent (default: 0.001).
- --gamma: Discount factor gamma for the DeepQAgent (default: 0.95).
- --epsilon: Epsilon value for the DeepQAgent epsilon-greedy action selection (default: 0.1).

## Playing Against the Agent
To play against the trained agent, be sure to include the following argument to play with it after training:
```bash
  python main.py --generate --human
```
Optional arguments:
- --human_episodes: Number of rounds to play against the agent when using the --human flag (default: 5).

## Results
After running the evaluation, you can find the results in the experiments directory. The following metrics are calculated:

- Win Rate
- Average Expected Earnings
- Action Entropy
- Initial Action Distribution (FOLD, CALL, RAISE, ALL IN)
- Average Initial Raise Fraction of Pot
- Additionally, convergence rates are plotted and saved in the experiments directory.

