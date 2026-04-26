"""Unit tests for the parking ABM.  Run with: pytest tests/"""

import pytest
from src.model import ParkingModel
from src.agents import SpaceAgent, VehicleAgent, ManagerAgent

SMALL = {"n_bays": 30, "n_vehicles": 10, "steps_per_hour": 60,
         "mean_park_duration": 4, "std_park_duration": 1}


@pytest.fixture
def agent_model():
    return ParkingModel(use_agent_system=True, config=SMALL, seed=0)


@pytest.fixture
def fcfs_model():
    return ParkingModel(use_agent_system=False, config=SMALL, seed=0)


# ── ManagerAgent ──────────────────────────────────────────────────────────────

def test_manager_registers_all_bays(agent_model):
    assert len(agent_model.manager_agent._bays) == SMALL["n_bays"]

def test_manager_allocate_returns_valid_type(agent_model):
    result = agent_model.manager_agent.allocate_bay((0, 0), "student")
    assert result is None or isinstance(result, int)

def test_manager_update_bay(agent_model):
    mgr = agent_model.manager_agent
    bid = next(iter(mgr._bays))
    mgr.update_bay(bid, occupied=True)
    assert mgr._bays[bid]["occupied"] is True
    mgr.update_bay(bid, occupied=False)
    assert mgr._bays[bid]["occupied"] is False


# ── SpaceAgent ────────────────────────────────────────────────────────────────

def test_bays_initially_free(agent_model):
    bays = [a for a in agent_model.schedule.agents if isinstance(a, SpaceAgent)]
    assert all(not b.occupied for b in bays)

def test_occupy_vacate(agent_model):
    bays = [a for a in agent_model.schedule.agents if isinstance(a, SpaceAgent)]
    b = bays[0]
    b.occupy(999)
    assert b.occupied
    b.vacate()
    assert not b.occupied


# ── VehicleAgent ──────────────────────────────────────────────────────────────

def test_vehicles_start_waiting(agent_model):
    vehicles = [a for a in agent_model.schedule.agents if isinstance(a, VehicleAgent)]
    assert all(v.state == VehicleAgent.WAITING for v in vehicles)


# ── ParkingModel ──────────────────────────────────────────────────────────────

def test_agent_model_runs(agent_model):
    agent_model.run(steps=20)

def test_fcfs_model_runs(fcfs_model):
    fcfs_model.run(steps=20)

def test_utilisation_in_range(agent_model):
    agent_model.run(steps=20)
    assert 0.0 <= agent_model.mean_utilisation <= 1.0

def test_search_times_non_negative(agent_model):
    agent_model.run(steps=20)
    assert all(t >= 0 for t in agent_model._search_times)

def test_agent_faster_than_fcfs():
    """Agent-based system should achieve lower mean search time than FCFS."""
    cfg = {"n_bays": 50, "n_vehicles": 20, "steps_per_hour": 60,
           "mean_park_duration": 4, "std_park_duration": 1}
    a = ParkingModel(use_agent_system=True,  config=cfg, seed=5)
    f = ParkingModel(use_agent_system=False, config=cfg, seed=5)
    a.run(); f.run()
    if a._search_times and f._search_times:
        assert a.mean_search_time <= f.mean_search_time, (
            f"Agent ({a.mean_search_time:.1f}s) should not exceed FCFS ({f.mean_search_time:.1f}s)"
        )
