# Configuration parameters for assets, RL hyperparameters, and environment settings

# Data
ASSETS = ['MSTF', 'JNJ', 'JPM', 'NVDA', 'PG', 'XOM']  
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
LOOKBACK_WINDOW = 252  # 1 year of trading days

# Environment
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001  # 0.1%
REBALANCE_FREQUENCY = 22  # 22 trading days - monthly rebalancing

# RL Configuration
STATE_DIM = len(ASSETS) * 4  # returns, volatility, correlation features, hrp weights
ACTION_DIM = len(ASSETS)  # portfolio weights
EPISODE_LENGTH = 252  # 1 year episodes
LEARNING_RATE = 1e-4
GAMMA = 0.95
BATCH_SIZE = 32
MEMORY_SIZE = 10000

# HRP Configuration
HRP_LOOKBACK = 60  # Days for HRP calculation