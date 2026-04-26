"""
Mesa ParkingModel for the Keele University campus.

Supports both conditions:
  use_agent_system=True  → ManagerAgent proximity-based allocation
  use_agent_system=False → FCFS baseline (sequential scan)

Time resolution: 1 step = 1 minute (steps_per_hour=60).
This gives realistic search times: FCFS vehicles typically take 2–4 steps
(120–240 s) to find a bay; the agent-based system reduces this to 1–2 steps.
"""

import random
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from src.agents import SpaceAgent, VehicleAgent, ManagerAgent

# Keele campus parameters – calibrated from public campus occupancy reports
DEFAULT_CONFIG = {
    "n_bays": 2000,
    "grid_width": 50,
    "grid_height": 40,
    "enforcement_start": 8,      # 08:00
    "enforcement_end": 18,       # 18:00
    "steps_per_hour": 60,        # 1 step = 1 minute
    "n_vehicles": 1000,          # daily vehicle visits (realistic for 2,000-bay campus)
    "mean_park_duration": 240,   # steps = 4 hours
    "std_park_duration": 60,     # ± 1 hour
    "permit_split": {
        "student": 0.60,
        "staff": 0.30,
        "visitor": 0.10,
    },
    "seed": 42,
}

# CO₂ estimation (EPA method, following Al-Khafajiy et al. 2020)
_KG_CO2_PER_GALLON = 8.89    # EPA conversion factor
_FUEL_EFFICIENCY_MPG = 25.0  # average vehicle fuel efficiency
_CRUISE_SPEED_KMH = 15.0     # typical parking lot cruising speed


class ParkingModel(Model):
    """
    Agent-based parking model.

    Parameters
    ----------
    use_agent_system : bool
        If True, ManagerAgent directs vehicles. If False, FCFS baseline.
    config : dict, optional
        Override DEFAULT_CONFIG values.
    seed : int, optional
        Random seed (for replication, following ter Hofstede et al. 2023).
    """

    def __init__(self, use_agent_system=True, config=None, seed=None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        if seed is not None:
            cfg["seed"] = seed
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        self.cfg = cfg
        self.use_agent_system = use_agent_system
        self.schedule = RandomActivation(self)
        self._search_times = []   # seconds, recorded when each vehicle parks
        self._uid = 0

        # --- ManagerAgent ---
        self.manager_agent = ManagerAgent(self._next_id(), self)
        self.schedule.add(self.manager_agent)

        # --- SpaceAgents (parking bays) ---
        permit_labels = list(cfg["permit_split"].keys())
        counts = [int(cfg["n_bays"] * cfg["permit_split"][k]) for k in permit_labels]
        counts[0] += cfg["n_bays"] - sum(counts)  # absorb rounding remainder

        permit_pool = []
        for label, count in zip(permit_labels, counts):
            permit_pool.extend([label] * count)
        random.shuffle(permit_pool)

        for i in range(cfg["n_bays"]):
            loc = (
                random.randint(0, cfg["grid_width"] - 1),
                random.randint(0, cfg["grid_height"] - 1),
            )
            bay = SpaceAgent(self._next_id(), self, location=loc,
                             permit_category=permit_pool[i])
            self.schedule.add(bay)
            self.manager_agent.register_bay(bay)

        # --- VehicleAgents ---
        total_steps = (cfg["enforcement_end"] - cfg["enforcement_start"]) * cfg["steps_per_hour"]
        for _ in range(cfg["n_vehicles"]):
            arrival = random.randint(0, total_steps - 1)
            duration = max(1, int(np.random.normal(cfg["mean_park_duration"],
                                                    cfg["std_park_duration"])))
            # vehicles enter from the perimeter
            if random.random() < 0.5:
                entry = (random.randint(0, cfg["grid_width"] - 1),
                         random.choice([0, cfg["grid_height"] - 1]))
            else:
                entry = (random.choice([0, cfg["grid_width"] - 1]),
                         random.randint(0, cfg["grid_height"] - 1))

            permit = random.choices(
                permit_labels,
                weights=[cfg["permit_split"][k] for k in permit_labels]
            )[0]

            vehicle = VehicleAgent(
                self._next_id(), self,
                arrival_step=arrival,
                parking_duration=duration,
                entry_location=entry,
                permit_category=permit,
                use_agent_system=use_agent_system,
            )
            self.schedule.add(vehicle)

        # --- DataCollector ---
        self.datacollector = DataCollector(
            model_reporters={
                "Utilisation": self._utilisation,
                "ParkedCount": lambda m: sum(
                    1 for a in m.schedule.agents
                    if isinstance(a, VehicleAgent) and a.state == VehicleAgent.PARKED
                ),
            }
        )

    def _next_id(self):
        uid = self._uid
        self._uid += 1
        return uid

    def record_search_time(self, steps):
        """Called by VehicleAgent when it successfully parks."""
        seconds = steps * (3600 / self.cfg["steps_per_hour"])
        self._search_times.append(seconds)

    def _utilisation(self):
        bays = [a for a in self.schedule.agents if isinstance(a, SpaceAgent)]
        if not bays:
            return 0.0
        return sum(1 for b in bays if b.occupied) / len(bays)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, steps=None):
        total = steps or (
            (self.cfg["enforcement_end"] - self.cfg["enforcement_start"])
            * self.cfg["steps_per_hour"]
        )
        for _ in range(total):
            self.step()

    @property
    def mean_search_time(self):
        """Mean search time in seconds across all vehicles that successfully parked."""
        return float(np.mean(self._search_times)) if self._search_times else 0.0

    @property
    def mean_utilisation(self):
        df = self.datacollector.get_model_vars_dataframe()
        return float(df["Utilisation"].mean()) if not df.empty else 0.0

    def co2_savings_vs_baseline(self, fcfs_mean_seconds):
        """
        Estimate CO₂ saved vs. a FCFS baseline run.
        Uses EPA conversion: 8.89 kg CO₂/gallon, 25 mpg, 15 km/h cruising.

        Parameters
        ----------
        fcfs_mean_seconds : float
            Mean search time of the FCFS baseline run (seconds).
        """
        if not self._search_times:
            return 0.0
        saved_per_vehicle = fcfs_mean_seconds - self.mean_search_time
        if saved_per_vehicle <= 0:
            return 0.0

        cruise_ms = _CRUISE_SPEED_KMH * 1000 / 3600          # m/s
        total_dist_saved_m = saved_per_vehicle * cruise_ms * len(self._search_times)
        dist_saved_miles = total_dist_saved_m / 1609.34
        fuel_saved_gallons = dist_saved_miles / _FUEL_EFFICIENCY_MPG
        return fuel_saved_gallons * _KG_CO2_PER_GALLON
