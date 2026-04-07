# import numpy as np
# import pandas as pd

# class MarketPriors:

#     def __init__(self):

#         # ---------------------------
#         # Demand level
#         # ---------------------------
#         self.alpha = 4.0
#         self.season_weight = 0.35

#         # Elasticity
#         self.elasticity = 4.5
#         self.season_elast_amp = 0.05
#         self.gamma = 0.01  # curvature

#         # Competition
#         self.cross_elasticity = 1.0
#         self.rel_price_weight = 1.3

#         # Shocks (AR(1))
#         self.rho = 0.6
#         self.shock_std = 0.15

#         # Competitor dynamics
#         self.comp_mean = 120
#         self.comp_noise = 10.0
#         self.comp_follow = 0.1
#         self.comp_reversion = 0.05

#         # Costs
#         self.unit_cost = 60
#         self.fixed_cost = 200

#         # Bounds
#         self.price_min = 100
#         self.price_max = 160

#         # Market size
#         self.max_demand = 500  # cap demand

# def seasonal_shape(phase):
#     return np.exp(-((phase - 0.1) ** 2) / 0.08)

# def payday_effect(t, rng, period=30):

#     phase = (t % period) / period
#     cycle = t // period

#     # store amplitudes in dict (persistent)
#     if not hasattr(payday_effect, "cycle_amplitudes"):
#         payday_effect.cycle_amplitudes = {}

#     if cycle not in payday_effect.cycle_amplitudes:
#         payday_effect.cycle_amplitudes[cycle] = rng.lognormal(mean=0.0, sigma=0.3)

#     amplitude = payday_effect.cycle_amplitudes[cycle]
#     return amplitude * np.exp(-((phase - 0.1) ** 2) / 0.08)

# def update_competitor_price(last_comp, agent_price, priors, rng):

#     # partial follow + own inertia + noise
#     comp_price = (
#         last_comp
#         + priors.comp_reversion * (priors.comp_mean - last_comp)
#         + priors.comp_follow * (agent_price - last_comp)
#         + rng.normal(0, priors.comp_noise)
#     )

#     return comp_price

# def compute_demand(price, comp_price, season, eta, priors):

#     log_price = np.log(price)
#     log_comp = np.log(comp_price)
#     rel_price = log_comp - log_price

#     base_log_mu = (priors.alpha + (priors.season_weight * season) + eta)
    
#     elasticity_t = priors.elasticity * np.exp(-priors.season_elast_amp * season)
#     price_term = ((- elasticity_t * log_price) + (- priors.gamma * (log_price ** 2)))

#     rel_price_term = priors.rel_price_weight * rel_price

#     log_mu = base_log_mu + price_term + rel_price_term
#     mu = np.exp(log_mu)
    
#     return np.clip(mu, 1, 2500)

# def demand_step(df_hist, policy_fn, priors, rng):

#     t = len(df_hist)
#     last_row = df_hist.iloc[-1]

#     season = payday_effect(t, rng)

#     # decision at time t
#     price = policy_fn(df_hist, priors, rng)
#     eta = priors.rho * last_row["demand_shock"] + rng.normal(0, priors.shock_std)
#     comp_price = update_competitor_price(last_row["price_competitor"], last_row["price_agent"], priors, rng)

#     # demand reacts to current settings
#     demand = compute_demand(price, comp_price, season, eta, priors)
#     revenue = (price - priors.unit_cost) * demand - priors.fixed_cost

#     return {
#         "price_agent": price,
#         "price_competitor": comp_price,
#         "season_signal": season,
#         "demand_shock": eta,
#         "realized_demand": demand,
#         "revenue": revenue
#     }

# def simulate(n_days, policy_fn, priors=MarketPriors, seed=42, start_df=None):

#     rng = np.random.default_rng(seed)

#     if start_df is None:

#         start_date = pd.Timestamp("2025-01-01")
        
#         df = pd.DataFrame([{
#             "date": start_date,
#             "price_agent": 130,
#             "price_competitor": 120,
#             "season_signal": payday_effect(0, rng),
#             "demand_shock": 0.0,
#             "realized_demand": 50,
#             "revenue": 0
#         }])

#     else:
#         df = start_df.copy()

#     # evolve environment 
#     for i in range(n_days):

#         new_price = policy_fn(df, priors, rng)
#         df.loc[df.index[-1], "price_agent"] = new_price

#         new_row = demand_step(df, policy_fn, priors, rng)
#         new_row["date"] = df.iloc[-1]["date"] + pd.Timedelta(days=1)

#         df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

#     return df

# def random_policy(df_hist, priors, rng):

#     return float(rng.uniform(priors.price_min, priors.price_max))

# =========================================================
# Stable Competitive Market Simulation
# =========================================================

import numpy as np
import pandas as pd


# =========================================================
# Priors
# =========================================================
class MarketPriors:

    def __init__(self):

        # Demand level
        self.alpha = 2.5
        self.season_weight = 0.6
        self.ref_price = 130

        # Elasticity (STRONG SIGNAL)
        self.elasticity = 3.5
        self.season_elast_amp = 0.4
        self.gamma = 0.15  # stronger curvature (critical)

        # Competition (STRONG)
        self.cross_elasticity = 1.2
        self.rel_price_weight = 1.5

        # Shocks
        self.rho = 0.6
        self.shock_std = 0.12

        # Competitor dynamics
        self.comp_mean = 120
        self.comp_noise = 6.0
        self.comp_follow = 0.25   # stronger reaction
        self.comp_reversion = 0.08

        # Costs
        self.unit_cost = 60
        self.fixed_cost = 200

        # Bounds
        self.price_min = 100
        self.price_max = 160

        # Market size (bounded demand)
        self.max_demand = 200


# =========================================================
# Seasonality
# =========================================================
def payday_effect(t, rng, period=30):

    phase = (t % period) / period
    cycle = t // period

    if not hasattr(payday_effect, "cycle_amplitudes"):
        payday_effect.cycle_amplitudes = {}

    if cycle not in payday_effect.cycle_amplitudes:
        payday_effect.cycle_amplitudes[cycle] = rng.lognormal(0.0, 0.25)

    amplitude = payday_effect.cycle_amplitudes[cycle]

    return amplitude * np.exp(-((phase - 0.1) ** 2) / 0.05)


# =========================================================
# Competitor dynamics
# =========================================================
def update_competitor_price(last_comp, agent_price, last_demand, priors, rng):

    # ---------------------------
    # 1. State proxy (observable)
    # ---------------------------
    demand_signal = np.log(last_demand + 1)

    # ---------------------------
    # 2. Core pricing logic
    # ---------------------------
    # competitor follows:
    # - own inertia
    # - your lagged price
    # - demand conditions

    base_price = (
        0.7 * last_comp                                # strong inertia
        + 0.2 * agent_price                            # reacts to you
        + 0.1 * priors.comp_mean                       # long-term anchor
    )

    # demand-based adjustment (proxy for market conditions)
    base_price += 2.0 * demand_signal

    # ---------------------------
    # 3. Controlled exploration (LOW)
    # ---------------------------
    u = rng.uniform()

    if u < 0.1:
        # rare global exploration
        price = rng.uniform(50, 200)

    else:
        # mostly local adjustment
        price = base_price + rng.normal(0, 3.0)

    # ---------------------------
    # 4. Small noise (always)
    # ---------------------------
    price += rng.normal(0, priors.comp_noise * 0.3)

    return float(np.clip(price, priors.price_min, priors.price_max))

# =========================================================
# Stable demand (NO EXPLOSION)
# =========================================================
def compute_demand(price, comp_price, season, shock, priors):

    log_price = np.log(price / priors.ref_price)
    log_comp = np.log(comp_price / priors.ref_price)

    # Relative price (important driver)
    rel_price = log_comp - log_price

    # State-dependent elasticity
    #elasticity_t = priors.elasticity * (1 - priors.season_elast_amp * season)
    elasticity_t = priors.elasticity * np.exp(-priors.season_elast_amp * season)

    # Smooth price transform (critical for stability)
    #price_smooth = np.tanh(log_price)

    # Latent score
    z = (
        priors.alpha
        + priors.season_weight * season
        + shock
        - elasticity_t * log_price
        - priors.gamma * (log_price ** 2)
        + priors.cross_elasticity * log_comp
        + priors.rel_price_weight * rel_price
    )

    # Bounded demand (FIX)
    demand = np.exp(z) # priors.max_demand / (1 + np.exp(-z))

    return float(np.clip(demand, 1.0, priors.max_demand))


# =========================================================
# One step
# =========================================================
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


# =========================================================
# Simulation
# =========================================================
def simulate(n_days, policy_fn, priors=None, seed=42, start_df=None):

    rng = np.random.default_rng(seed)

    if priors is None:
        priors = MarketPriors()

    if start_df is None:

        start_date = pd.Timestamp("2025-01-01")

        demand = 50.0
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

    t = len(df_hist)
    last = df_hist.iloc[-1]

    # ---------------------------
    # 1. Observable proxy for state
    # ---------------------------
    # Only use what agent "sees"
    demand_proxy = last["realized_demand"]
    comp_price = last["price_competitor"]

    # Normalize demand proxy (stabilizes scale)
    demand_norm = np.log(demand_proxy + 1)

    # ---------------------------
    # 2. Anchor price (competitor-aware)
    # ---------------------------
    # follow competitor + mean reversion
    base_price = (
        0.6 * comp_price
        + 0.4 * priors.ref_price
    )

    # demand-driven adjustment (proxy for seasonality)
    base_price += 3.0 * demand_norm

    # ---------------------------
    # 3. Exploration regimes
    # ---------------------------
    u = rng.uniform()

    if u < 0.2:
        # GLOBAL exploration
        price = rng.uniform(priors.price_min, priors.price_max)

    elif u < 0.7:
        # LOCAL exploration (around anchor)
        price = base_price + rng.normal(0, 6.0)

    else:
        # STRUCTURED sweep (ensures identification)
        cycle_pos = (t % 25) / 25
        price = priors.price_min + cycle_pos * (priors.price_max - priors.price_min)

    # ---------------------------
    # 4. Small noise (break determinism)
    # ---------------------------
    price += rng.normal(0, 1.5)

    return float(np.clip(price, priors.price_min, priors.price_max))