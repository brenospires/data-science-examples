import numpy as np
import pandas as pd

class MarketPriors:

    def __init__(self):

        # demand level
        self.alpha = 17.0          # base log demand
        self.season_weight = 0.35  # payday boost

        #  elasticity 
        self.elasticity = 2.5
        self.season_elast_amp = 0.05 # reduces elasticity at peaks
        self.gamma = 0.01 #  price curvature 

        # competition
        self.cross_elasticity = 1.0
        self.rel_price_weight = 0.3

        # shocks
        self.rho = 0.6
        self.shock_std = 0.15

        # competitor dynamics
        self.comp_mean = 120
        self.comp_noise = 10.0
        self.comp_follow = 0.0
        self.comp_reversion = 0.05
        
        # costs
        self.unit_cost = 60
        self.fixed_cost = 200

        # bounds
        self.price_min = 100
        self.price_max = 160

def seasonal_shape(phase):
    return np.exp(-((phase - 0.1) ** 2) / 0.08)

def payday_effect(t, rng, period=30):

    phase = (t % period) / period
    cycle = t // period

    # store amplitudes in dict (persistent)
    if not hasattr(payday_effect, "cycle_amplitudes"):
        payday_effect.cycle_amplitudes = {}

    if cycle not in payday_effect.cycle_amplitudes:
        payday_effect.cycle_amplitudes[cycle] = rng.lognormal(mean=0.0, sigma=0.3)

    amplitude = payday_effect.cycle_amplitudes[cycle]
    return amplitude * np.exp(-((phase - 0.1) ** 2) / 0.08)

def update_competitor_price(last_comp, agent_price, priors, rng):

    # partial follow + own inertia + noise
    comp_price = (
        last_comp
        + priors.comp_reversion * (priors.comp_mean - last_comp)
        + priors.comp_follow * (agent_price - last_comp)
        + rng.normal(0, priors.comp_noise)
    )

    return comp_price

def compute_demand(price, comp_price, season, eta, priors):

    log_price = np.log(price)
    log_comp = np.log(comp_price)
    rel_price = log_comp - log_price

    base_log_mu = (priors.alpha + (priors.season_weight * season) + eta)
    
    elasticity_t = priors.elasticity * np.exp(-priors.season_elast_amp * season)
    price_term = ((- elasticity_t * log_price) + (- priors.gamma * (log_price ** 2)))

    rel_price_term = priors.rel_price_weight * rel_price

    log_mu = base_log_mu + price_term + rel_price_term
    mu = np.exp(log_mu)
    
    return np.clip(mu, 1, 2500)

def demand_step(df_hist, policy_fn, priors, rng):

    t = len(df_hist)
    last_row = df_hist.iloc[-1]

    season = payday_effect(t, rng)

    # decision at time t
    price = policy_fn(df_hist, priors, rng)
    eta = priors.rho * last_row["demand_shock"] + rng.normal(0, priors.shock_std)
    comp_price = update_competitor_price(last_row["price_competitor"], last_row["price_agent"], priors, rng)

    # demand reacts to current settings
    demand = compute_demand(price, comp_price, season, eta, priors)
    revenue = (price - priors.unit_cost) * demand - priors.fixed_cost

    return {
        "price_agent": price,
        "price_competitor": comp_price,
        "season_signal": season,
        "demand_shock": eta,
        "realized_demand": demand,
        "revenue": revenue
    }

def simulate(n_days, policy_fn, priors=MarketPriors, seed=42, start_df=None):

    rng = np.random.default_rng(seed)

    if start_df is None:

        start_date = pd.Timestamp("2025-01-01")
        
        df = pd.DataFrame([{
            "date": start_date,
            "price_agent": 130,
            "price_competitor": 120,
            "season_signal": payday_effect(0, rng),
            "demand_shock": 0.0,
            "realized_demand": 50,
            "revenue": 0
        }])

    else:
        df = start_df.copy()

    # evolve environment 
    for i in range(n_days):

        new_price = policy_fn(df, priors, rng)
        df.loc[df.index[-1], "price_agent"] = new_price

        new_row = demand_step(df, policy_fn, priors, rng)
        new_row["date"] = df.iloc[-1]["date"] + pd.Timedelta(days=1)

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df

def random_policy(df_hist, priors, rng):

    return float(rng.uniform(priors.price_min, priors.price_max))