"""
Three cooperating agent types for the Keele University campus parking simulation.

  ManagerAgent  – supervisory; allocates the nearest free bay to incoming vehicles
  SpaceAgent    – one per parking bay; tracks occupancy and notifies the manager
  VehicleAgent  – simulates individual driver behaviour (arrive, search, park, leave)

Architecture follows Wooldridge (2009) hybrid MAS pattern. Inter-agent messaging
uses a simplified FIPA-ACL request/inform pattern (see Supporting Material §1.1).
"""

from mesa import Agent


class SpaceAgent(Agent):
    """Represents a single parking bay. Passive – mutated by VehicleAgent/ManagerAgent."""

    def __init__(self, unique_id, model, location, permit_category="student"):
        super().__init__(unique_id, model)
        self.location = location            # (x, y) grid coords
        self.permit_category = permit_category
        self.occupied = False
        self._vehicle_id = None

    def occupy(self, vehicle_id):
        self.occupied = True
        self._vehicle_id = vehicle_id
        self.model.manager_agent.update_bay(self.unique_id, occupied=True)

    def vacate(self):
        self.occupied = False
        self._vehicle_id = None
        self.model.manager_agent.update_bay(self.unique_id, occupied=False)

    def step(self):
        pass  # reactive only


class ManagerAgent(Agent):
    """
    Keeps a global occupancy map and allocates the nearest available
    permit-matching bay to incoming vehicles (proximity heuristic).
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self._bays = {}        # bay_id -> {location, occupied, permit}
        self._bay_agents = {}  # bay_id -> SpaceAgent ref

    def register_bay(self, bay_agent):
        bid = bay_agent.unique_id
        self._bays[bid] = {
            "location": bay_agent.location,
            "occupied": False,
            "permit": bay_agent.permit_category,
        }
        self._bay_agents[bid] = bay_agent

    def update_bay(self, bay_id, occupied):
        if bay_id in self._bays:
            self._bays[bay_id]["occupied"] = occupied

    def allocate_bay(self, vehicle_location, permit_category):
        """Return id of nearest free matching bay, or None if none available."""
        best_id = None
        best_dist = float("inf")
        vx, vy = vehicle_location

        for bid, info in self._bays.items():
            if info["occupied"] or info["permit"] != permit_category:
                continue
            bx, by = info["location"]
            dist = ((vx - bx) ** 2 + (vy - by) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = bid

        return best_id

    def step(self):
        pass


class VehicleAgent(Agent):
    """
    Simulates one vehicle/driver.

    Agent-based mode: requests directed allocation from ManagerAgent.
    FCFS baseline: scans bays sequentially until one is found.
    """

    WAITING = "waiting"
    SEARCHING = "searching"
    PARKED = "parked"
    DEPARTED = "departed"

    def __init__(self, unique_id, model, arrival_step, parking_duration,
                 entry_location, permit_category="student", use_agent_system=True):
        super().__init__(unique_id, model)
        self.arrival_step = arrival_step
        self.parking_duration = parking_duration
        self.entry_location = entry_location
        self.permit_category = permit_category
        self.use_agent_system = use_agent_system

        self.state = self.WAITING
        self.search_start = None
        self.search_time_steps = 0
        self.allocated_bay_id = None
        self.parked_since = None

    def step(self):
        tick = self.model.schedule.steps

        if self.state == self.WAITING:
            if tick >= self.arrival_step:
                self.state = self.SEARCHING
                self.search_start = tick

        elif self.state == self.SEARCHING:
            self._try_park(tick)

        elif self.state == self.PARKED:
            if tick >= self.parked_since + self.parking_duration:
                self._depart()

    def _try_park(self, tick):
        if self.use_agent_system:
            bay_id = self.model.manager_agent.allocate_bay(
                self.entry_location, self.permit_category
            )
        else:
            bay_id = self._fcfs_scan()

        if bay_id is not None:
            bay = self.model.manager_agent._bay_agents.get(bay_id)
            if bay and not bay.occupied:
                bay.occupy(self.unique_id)
                self.allocated_bay_id = bay_id
                self.search_time_steps = tick - self.search_start
                self.parked_since = tick
                self.state = self.PARKED
                self.model.record_search_time(self.search_time_steps)

    def _fcfs_scan(self):
        """First-come-first-served: take the first available matching bay."""
        for bay in self.model.manager_agent._bay_agents.values():
            if not bay.occupied and bay.permit_category == self.permit_category:
                return bay.unique_id
        return None

    def _depart(self):
        if self.allocated_bay_id is not None:
            bay = self.model.manager_agent._bay_agents.get(self.allocated_bay_id)
            if bay:
                bay.vacate()
        self.state = self.DEPARTED
