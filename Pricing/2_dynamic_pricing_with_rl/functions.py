
import numpy as np
import pandas as pd
import xgboost as xgb

class MarketPriors:

    def __init__(self):

        # Demand level
        self.alpha = 1.8          
        self.season_weight = 0.9  
        self.ref_price = 130

        # Elasticity
        self.elasticity = 2.5     
        self.season_elast_amp = 0.5  
        self.gamma = 0.2  

        # Competition 
        self.cross_elasticity = 1.1   
        self.rel_price_weight = 1.3   

        # Shocks
        self.rho = 0.6
        self.shock_std = 0.10  

        # Competitor dynamics
        self.comp_mean = 120
        self.comp_noise = 5.0  
        self.comp_follow = 0.20 
        self.comp_reversion = 0.08

        # Costs
        self.unit_cost = 60
        self.fixed_cost = 200

        # Bounds
        self.price_min = 100
        self.price_max = 160

        # Market size (bounded demand)
        self.max_demand = 200

# Seasonality
def payday_effect(t, rng, period=30):

    phase = (t % period) / period
    cycle = t // period

    if not hasattr(payday_effect, "cycle_amplitudes"):
        payday_effect.cycle_amplitudes = {}

    if cycle not in payday_effect.cycle_amplitudes:
        payday_effect.cycle_amplitudes[cycle] = rng.lognormal(0.0, 0.25)

    amplitude = payday_effect.cycle_amplitudes[cycle]
    return amplitude * np.exp(-((phase - 0.1) ** 2) / 0.05)

def update_competitor_price(last_comp, agent_price, last_demand, priors, rng):

    demand_signal = np.log(last_demand + 1)

    base_price = (
        0.85 * last_comp
        + 0.05 * agent_price
        + 0.10 * priors.comp_mean
    )

    base_price += 1.2 * demand_signal

    # multiplicative noise 
    sigma = 0.25
    price = base_price * np.exp(rng.normal(0, sigma))

    # rare exploration
    if rng.uniform() < 0.05:
        price = rng.uniform(priors.price_min, priors.price_max)

    return float(np.clip(price, priors.price_min, priors.price_max))

def compute_demand(price, comp_price, season, shock, priors):

    log_price = np.log(price / priors.ref_price)
    log_comp = np.log(comp_price / priors.ref_price)

    rel_price = log_comp - log_price

    # State-dependent elasticity
    elasticity_t = priors.elasticity * np.exp(-priors.season_elast_amp * season)

    z = (
        priors.alpha
        + priors.season_weight * season
        + shock
        - elasticity_t * log_price
        - priors.gamma * (log_price ** 2)
        + priors.cross_elasticity * log_comp
        + priors.rel_price_weight * rel_price
    )

    demand = np.exp(z)
    return float(np.clip(demand, 1.0, priors.max_demand))

def demand_step(df_hist, policy_fn, priors, rng):

    t = len(df_hist)
    last_row = df_hist.iloc[-1]

    season = payday_effect(t, rng)

    # policy decides price
    price = float(policy_fn(df_hist, priors, rng))

    # shock (AR(1))
    shock = (
        priors.rho * last_row["demand_shock"]
        + rng.normal(0, priors.shock_std)
    )

    # competitor reacts to LAST agent price (important timing fix)
    comp_price = update_competitor_price(last_row["price_competitor"], last_row["price_agent"], 
        last_row["realized_demand"], priors, rng)

    # demand realization
    demand = compute_demand(price, comp_price, season, shock, priors)

    # profit (not revenue)
    profit = (price - priors.unit_cost) * demand - priors.fixed_cost

    return {
        "price_agent": price,
        "price_competitor": comp_price,
        "season_signal": season,
        "demand_shock": shock,
        "realized_demand": demand,
        "revenue": profit
    }

def simulate(n_days, policy_fn, priors=None, seed=42, start_df=None):

    rng = np.random.default_rng(seed)

    if priors is None:
        priors = MarketPriors()

    if start_df is None:

        start_date = pd.Timestamp("2025-01-01")

        demand = 25.0
        revenue = (130.0 - priors.unit_cost) * demand - priors.fixed_cost

        df = pd.DataFrame([{
            "date": start_date,
            "price_agent": 130.0,
            "price_competitor": 120.0,
            "season_signal": payday_effect(0, rng),
            "demand_shock": 0.0,
            "realized_demand": demand,
            "revenue": revenue
        }])

        n_days -= 1

    else:
        df = start_df.copy()

    if isinstance(policy_fn, BanditPolicy):
        X = []

        for i in range(1, len(df)):
            state = extract_state(df.iloc[:i+1])
            price = df.iloc[i]["price_agent"]
            phi = build_features(state, price)
            X.append(phi)

        if len(X) > 0:
            policy_fn.scaler.fit(np.array(X))

    for t in range(n_days):

        new_row = demand_step(df, policy_fn, priors, rng)

        if isinstance(policy_fn, BanditPolicy):

            revenue = new_row["revenue"]
            price = new_row["price_agent"]
            reward = np.sign(revenue) * np.log1p(abs(revenue))
            
            #reward = np.log(new_row["realized_demand"])

            new_row["reward"] = reward
            policy_fn.update(df, price, reward)

        new_row["date"] = df.iloc[-1]["date"] + pd.Timedelta(days=1)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df

def exploration_policy(df_hist, priors, rng):

    last = df_hist.iloc[-1]

    demand_proxy = last["realized_demand"]
    comp_price = last["price_competitor"]

    demand_norm = np.log(demand_proxy + 1)

    # base anchor
    base_price = 0.2 * comp_price + 0.8 * priors.ref_price
    base_price += 2.0 * demand_norm

    sigma = 0.4   # controls log variance (~0.16)
    price = base_price * np.exp(rng.normal(0, sigma))

    # occasional regime jump
    if rng.uniform() < 0.15:
        price = rng.uniform(priors.price_min, priors.price_max)

    return float(np.clip(price, priors.price_min, priors.price_max))

def safe_log(x):
    EPS = 1e-6
    return np.log(np.maximum(x, EPS))

def build_features(state, price):
    
    # Extract state
    lag_demand = state["lag_1"]
    lag_price = state["price_lag_1"]
    lag_comp = state["competitor_lag_1"]
    comp_ma_7 = state["competitor_ma_7"]

    # proxy for current competition
    comp_est = 0.3 * lag_comp + 0.7 * comp_ma_7

    log_lag_demand = safe_log(lag_demand)
    log_lag_price = safe_log(lag_price)
    log_comp_est = safe_log(comp_est)
    log_lag_comp = safe_log(lag_comp)
    log_price = safe_log(price)
    
    log_rel_price_lag = log_lag_price - log_lag_comp
    log_rel_price = log_price - log_comp_est  
    price_change = log_price - log_lag_price 

    priors = MarketPriors()
    margin = price - priors.unit_cost

    features = np.array([
        # Context
        log_lag_demand,         # latent demand proxy
        log_lag_price,          # past decision
        log_comp_est,           # competitor proxy
        log_lag_comp,           
        log_rel_price_lag,      # past positioning

        # Action
        log_price,
        log_price ** 2,

        price,
        price ** 2,
        price ** 3,

        # Interactions
        log_rel_price,          # current positioning (approx)
        price_change,           # how aggressive current move is

        # Margin ffeatures
        margin,
        margin * log_lag_demand,
        margin * log_rel_price,

    ])

    return features

def extract_state(df):
    last = df.iloc[-1]

    if len(df) > 1:
        prev = df.iloc[-2]

        lag_demand = prev["realized_demand"]
        lag_price = prev["price_agent"]
        lag_comp = prev["price_competitor"]

        # rolling mean without full recomputation
        if len(df) > 7:
            comp_ma_7 = df["price_competitor"].iloc[-8:-1].mean()
        else:
            comp_ma_7 = lag_comp
    else:
        lag_demand = last["realized_demand"]
        lag_price = last["price_agent"]
        lag_comp = last["price_competitor"]
        comp_ma_7 = lag_comp

    return {
        "lag_1": lag_demand,
        "price_lag_1": lag_price,
        "competitor_lag_1": lag_comp,
        "competitor_ma_7": comp_ma_7,
    }

class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        X: array of shape (n_samples, n_features)
        """
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # avoid division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LinUCB:
    def __init__(self, n_features, alpha=1.0):
        self.alpha = alpha

        self.A = np.eye(n_features)        # (d x d)
        self.b = np.zeros(n_features)      # (d,)

    def get_theta(self):
        A_inv = np.linalg.inv(self.A)
        return A_inv @ self.b, A_inv
    
    def select_action(self, state, price_grid, build_features, scaler):
        
        theta, A_inv = self.get_theta()
        best_score = -np.inf
        best_price = None

        for price in price_grid:
            phi_raw = build_features(state, price)
            phi = scaler.transform(phi_raw)
            phi = np.clip(phi, -5, 5)

            mean = theta @ phi
            uncertainty = self.alpha * np.sqrt(phi @ A_inv @ phi)

            score = mean + uncertainty

            if score > best_score:
                best_score = score
                best_price = price

        return best_price

class LinearThompson:
    def __init__(self, n_features, v=1.0):
        self.v = v  # exploration scale

        self.A = np.eye(n_features)
        self.b = np.zeros(n_features)

    def sample_theta(self):
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b

        # sample from multivariate normal
        theta_sample = np.random.multivariate_normal(mu, self.v**2 * A_inv)

        return theta_sample

    def update(self, phi, reward):
        self.A += np.outer(phi, phi)
        self.b += reward * phi

class BanditPolicy:
    def __init__(self, agent, price_grid, scaler):
        self.agent = agent
        self.price_grid = price_grid
        self.scaler = scaler

    def __call__(self, df_hist, priors, rng):
        state = extract_state(df_hist)

        theta = self.agent.sample_theta()  

        best_price = None
        best_score = -np.inf

        for price in self.price_grid:

            phi_raw = build_features(state, price)
            phi = self.scaler.transform(phi_raw)
            phi = np.clip(phi, -5, 5)

            score = theta @ phi 

            if score > best_score:
                best_score = score
                best_price = price

        return float(np.clip(best_price, priors.price_min - 10, priors.price_max + 10))

    def update(self, df_hist, price, reward):
        state = extract_state(df_hist)

        phi_raw = build_features(state, price)
        phi = self.scaler.transform(phi_raw)
        phi = np.clip(phi, -5, 5)

        self.agent.update(phi, reward)
