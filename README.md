# Stock Market Trading Agent using Deep Reinforcement Learning

A Deep Q-Learning-based stock trading agent that learns to buy, sell, or hold stocks using historical data and technical indicators, built with PyTorch, Gym, and custom financial environments.

## Project Overview

This project implements a Dueling Deep Q-Network (DQN) for algorithmic trading, trained on stock market data. It simulates a trading environment using OpenAI Gym, incorporates technical indicators like RSI, MACD, and Bollinger Bands, and optimizes the trading policy to maximize portfolio value while minimizing risk using the Sharpe Ratio.

## Features

- Custom OpenAI Gym trading environment with configurable portfolio, trading strategy, and market window
- Dueling DQN architecture for stable and efficient policy learning
- Rolling-window training for robust generalization across market regimes
- Reward function incorporating profit percentage, trade penalties, and Sharpe ratio
- Visual analytics including reward curves, portfolio growth, and Sharpe ratio tracking

## Technologies Used

- Python, PyTorch, NumPy, Pandas, Matplotlib
- OpenAI Gym (custom environment)
- TA-Lib / ta (technical indicators)
- Dueling DQN architecture
- Reinforcement Learning algorithms

## How It Works

1. *Preprocessing*: Load and normalize historical stock data with added indicators (RSI, MACD, Bollinger Bands).
2. *Environment*: A custom StocksEnv simulates the portfolio, balance, and trading interactions.
3. *Model*: A DuelingDQN neural network estimates action-value functions (Q-values).
4. *Training*: The agent is trained over multiple episodes using epsilon-greedy policy, target networks, and experience replay.
5. *Evaluation*: Rewards, Sharpe Ratio, and portfolio value are plotted per episode and rolling windows.

## Results

- Best observed Sharpe Ratio: *9.12*
- Portfolio consistently maintained or grew over rolling training windows
- Significant improvement over random or static policies

### Display Output:

- Design a user-friendly interface to display classification results on a connected display.
- Present captured images along with predicted digits and confidence scores for improved usability.

<table>
  <tr>
    <td>
      <img src="Outputs/sharpe_ratio.png" alt="Setup" width="400"/>
    </td>
    <td>
      <img src="Outputs/equity_curve.png" alt="Setup" width="400"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="Outputs/price_curve.png" alt="Setup" width="400"/>
    </td>
    <td>
      <img src="Outputs/model_comp.png" alt="Setup" width="400"/>
    </td>
  </tr>
</table>
