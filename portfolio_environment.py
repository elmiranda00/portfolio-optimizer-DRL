import gym
from gym import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Any
from hrp_optimizer import optimizeHRP

class PortfolioEnv(gym.Env):
    """Provides the environment for DRL portfolio optimization with HRP baseline"""
    
    def __init__(self, config):
        super().__init__()
        
        # Get environment variables
        self.assets = config['assets']
        self.initial_capital = config['initial_capital']
        self.transaction_cost = config['transaction_cost']
        self.lookback_window = config['lookback_window']
        self.hrp_lookback = config['hrp_lookback']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        
        # Load data
        self.data = self._load_data()
        self.returns = self.data.pct_change().dropna()
        
        # Initialize HRP optimizer
        self.hrp_optimizer = optimizeHRP()
        
        # Action Space: agent that outputs a vector of portfolio weights between 0 and 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(self.assets),), dtype=np.float32
        )
        # Obs Space: state vector with returns, volatility, correlations, and HRP weights
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.assets) * 4,), dtype=np.float32
        )
        
        self.portfolio_returns_history = []
        self.reset()
    

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(len(self.assets)) / len(self.assets)
        return self._get_state()
    

    def _load_data(self) -> pd.DataFrame:
        """Load asset price data"""
        
        # Get adjusted close prices for all assets and make it a DataFrame
        data = yf.download(self.assets, start=self.start_date, end=self.end_date)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Fill missing values (holidays, weekends)
        data = data.ffill().bfill()
    
        return data
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        
        if self.current_step < self.lookback_window:
            # Pad with zeros if insufficient history
            recent_returns = np.zeros((self.lookback_window, len(self.assets)))
        else:
            recent_returns = self.returns.iloc[
                self.current_step - self.lookback_window:self.current_step
            ].values
        
        # Calculate features
        mean_returns = np.mean(recent_returns, axis=0) # mean return
        volatility = np.std(recent_returns, axis=0) # volatility
        corr_matrix = np.corrcoef(recent_returns.T)
        corr_features = corr_matrix[np.triu_indices_from(corr_matrix, k=1)] # corr. matrix
        
        if self.current_step >= self.hrp_lookback:
            hrp_returns = recent_returns[-self.hrp_lookback:]
            hrp_weights = self.hrp_optimizer.optimize(hrp_returns) # baseline HRP weights
        else:
            hrp_weights = np.ones(len(self.assets)) / len(self.assets)
        
        # Return features
        state = np.concatenate([
            mean_returns,
            volatility,
            corr_features[:len(self.assets)],  # Truncate to maintain fixed size
            hrp_weights
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Ensure post action that weights sum to 1
        action = action / (np.sum(action) + 1e-8)
        
        # A. Calculate portfolio return
        if self.current_step >= len(self.returns):
            done = True
            return self._get_state(), 0, done, {}
        period_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.sum(action * period_returns) # Weighted sum of individual asset returns
        
        # B. Calculate transaction costs
        portfolio_change = np.sum(np.abs(action - self.current_weights))
        transaction_costs = portfolio_change * self.transaction_cost
        
        # Net return (A - B)
        net_return = portfolio_return - transaction_costs
        
        # Update portfolio value and weights
        self.portfolio_value *= (1 + net_return)
        self.current_weights = action
        self.current_step += 1
        
        # Calculate reward (Sharpe ratio)
        
        self.portfolio_returns_history.append(net_return) # Save current return for Sharpe ratio

        returns_array = np.array(self.portfolio_returns_history[-self.lookback_window:]) # Calculate Sharpe ratio over last N steps
        if len(returns_array) > 1:
            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)
            reward = mean_return / (volatility + 1e-8)  # Sharpe ratio
        else:
            reward = 0.0
        
        reward = np.clip(reward, -5, 5) * 100 # clip extreme values and scale reward for training stability
        done = self.current_step >= len(self.returns) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': net_return,
            'transaction_costs': transaction_costs
        }
        
        return self._get_state(), reward, done, info
    
    