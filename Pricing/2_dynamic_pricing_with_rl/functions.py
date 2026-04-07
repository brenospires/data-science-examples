
import numpy as np
import pandas as pd

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

    else:
        df = start_df.copy()

    for _ in range(n_days - 1):

        new_row = demand_step(df, policy_fn, priors, rng)
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

