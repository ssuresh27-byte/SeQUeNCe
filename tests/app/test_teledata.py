# test_teledata.py

import os
import math
import itertools
import numpy as np
import pytest
from sequence.topology.dqc_net_topo import DQCNetTopo
from sequence.app.teleportation import TeledataApp
from sequence.kernel.quantum_utils import verify_same_state_vector

# Configs live next to this test file (tests/app/), so the suite is
# self-contained and runs regardless of the current working directory.
_CFG = os.path.dirname(os.path.abspath(__file__))

MILLISECOND = 1_000_000_000

# Deterministic random inputs and seeds for reproducible 5x5 grids
_rng = np.random.default_rng(2025)

def _random_state(rng: np.random.Generator) -> np.ndarray:
    # random complex 2-vector, normalized
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    vec = vec.astype(complex)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1, 0], dtype=complex)
    return vec / norm

RANDOM_PSIS   = [_random_state(_rng) for _ in range(5)]
SINGLE_SEEDS  = [
    {
        "alice": int(_rng.integers(0, 2**31-1)),
        "bob": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_bob": int(_rng.integers(0, 2**31-1)),
    }
    for _ in range(5)
]

def _all_nodes(topo):
    try:
        groups = topo.nodes.values()
        flat = []
        for g in groups:
            flat.extend(list(g))
        return flat
    except Exception:
        try:
            return list(topo.nodes)
        except Exception:
            return []


def single_trial(psi: np.ndarray, seeds=None):
    # set up the 2-node network
    topo = DQCNetTopo(os.path.join(_CFG, "teleport_2node.json"))
    tl   = topo.tl

    nodes = _all_nodes(topo)
    alice = next(n for n in nodes if getattr(n, 'name', '')=="alice")
    bob   = next(n for n in nodes if getattr(n, 'name', '')=="bob")
    bsm_nodes = [n for n in nodes if 'BSM' in getattr(n, 'name', '').upper()]
    bsm_ab = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_bob"), None)

    # set seeds
    if "alice" in seeds:
        try: alice.set_seed(seeds["alice"]) 
        except Exception: pass
    if "bob" in seeds:
        try: bob.set_seed(seeds["bob"]) 
        except Exception: pass
    if bsm_ab is not None and "BSM_alice_bob" in seeds:
        try: bsm_ab.set_seed(seeds["BSM_alice_bob"]) 
        except Exception: pass

    # 1) Prepare |ψ> in Alice’s data qubit
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    data_arr[0].update_state(psi.astype(complex))

    # 2) Attach the TeleportApp on both nodes
    A = TeledataApp(alice)
    B = TeledataApp(bob)

    # 3) Kick off teleport
    A.start(
        responder   = bob.name,
        start_t     = 10  * MILLISECOND,
        end_t       = 30 * MILLISECOND,
        memory_size = 1,
        fidelity    = 0.8,
        data_src    = 0
    )

    # 4) Run the simulation
    tl.init()
    tl.run()

    # 5) Read out Bob's data qubit state directly from the data memory
    data_arr = bob.get_component_by_name(bob.data_memo_arr_name)
    data_key = data_arr[0].qstate_key  # Check slot 0 (where the teleported state should be)
    full_state = tl.quantum_manager.get(data_key).state
    
    # Extract single-qubit state from potentially multi-qubit state
    if hasattr(full_state, "__len__") and len(full_state) > 2:
        # If it's a 2-qubit state, extract the first qubit (assuming it's the data qubit)
        half = len(full_state) // 2
        teleported_state = full_state[:half]
    else:
        teleported_state = full_state
    
    return np.array(teleported_state)

@pytest.mark.parametrize(
    "psi,seeds",
    list(itertools.product(RANDOM_PSIS, SINGLE_SEEDS))
)
def test_teledata_single_randomized(psi, seeds):
    out = single_trial(psi, seeds)
    assert out.shape == psi.shape
    assert verify_same_state_vector(out, psi)


def dual_trial(psi0: np.ndarray, psi1: np.ndarray, seeds=None):
    # Build 3-node topology (Alice, Bob, Charlie)
    topo = DQCNetTopo(os.path.join(_CFG, "teleport_3node.json"))
    tl   = topo.tl

    nodes = _all_nodes(topo)
    alice   = next(n for n in nodes if getattr(n, 'name', '')=="alice")
    bob     = next(n for n in nodes if getattr(n, 'name', '')=="bob")
    charlie = next(n for n in nodes if getattr(n, 'name', '')=="charlie")
    bsm_nodes = [n for n in nodes if 'BSM' in getattr(n, 'name', '').upper()]
    bsm_ab = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_bob"), None)
    bsm_ac = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_charlie"), None)

    # set seeds
    if "alice" in seeds:
        try: alice.set_seed(seeds["alice"]) 
        except Exception: pass
    if "bob" in seeds:
        try: bob.set_seed(seeds["bob"]) 
        except Exception: pass
    if "charlie" in seeds:
        try: charlie.set_seed(seeds["charlie"]) 
        except Exception: pass
    if bsm_ab is not None and "BSM_alice_bob" in seeds:
        try: bsm_ab.set_seed(seeds["BSM_alice_bob"]) 
        except Exception: pass
    if bsm_ac is not None and "BSM_alice_charlie" in seeds:
        try: bsm_ac.set_seed(seeds["BSM_alice_charlie"]) 
        except Exception: pass

    # Prepare two data qubits on Alice at indices 0 and 1
    a_key0 = alice.components[alice.data_memo_arr_name].memories[0].qstate_key
    a_key1 = alice.components[alice.data_memo_arr_name].memories[1].qstate_key
    alice.timeline.quantum_manager.set([a_key0], psi0.astype(complex))
    alice.timeline.quantum_manager.set([a_key1], psi1.astype(complex))

    # Attach apps on all participants
    A = TeledataApp(alice)
    B = TeledataApp(bob)
    C = TeledataApp(charlie)

    # Launch two concurrent teledata sessions: Alice→Bob (idx 0) and Alice→Charlie (idx 1)
    start_t1 = 1 * MILLISECOND
    end_t1   = 200   * MILLISECOND
    start_t2 = 1 * MILLISECOND
    end_t2   = 200   * MILLISECOND
    fidelity = 0.1
    mem_size = 1

    A.start(responder=bob.name, start_t=start_t1, end_t=end_t1,
            memory_size=mem_size, fidelity=fidelity, data_src=0)
    A.start(responder=charlie.name, start_t=start_t2, end_t=end_t2,
            memory_size=mem_size, fidelity=fidelity, data_src=1)

    # Run
    tl.init()
    tl.run()

    # Collect Bob and Charlie results by checking their data qubits directly
    # Bob should have the teleported state in slot 0, Charlie in slot 0 (wrapped due to modulo)
    bob_data_arr = bob.get_component_by_name(bob.data_memo_arr_name)
    charlie_data_arr = charlie.get_component_by_name(charlie.data_memo_arr_name)
    
    bob_data_key = bob_data_arr[0].qstate_key  # Slot 0
    charlie_data_key = charlie_data_arr[0].qstate_key  # Slot 0 (1 % 1 = 0)
    
    # Extract single-qubit states from potentially multi-qubit states
    def extract_single_qubit_state(full_state):
        if hasattr(full_state, "__len__") and len(full_state) > 2:
            half = len(full_state) // 2
            return full_state[:half]
        else:
            return full_state
    
    bob_state = extract_single_qubit_state(tl.quantum_manager.get(bob_data_key).state)
    charlie_state = extract_single_qubit_state(tl.quantum_manager.get(charlie_data_key).state)
    
    outs_b = [np.array(bob_state, dtype=complex)]
    outs_c = [np.array(charlie_state, dtype=complex)]
    return outs_b, outs_c


_dual_inputs = [(_random_state(_rng), _random_state(_rng)) for _ in range(5)]
_dual_seeds  = [
    {
        "alice": int(_rng.integers(0, 2**31-1)),
        "bob": int(_rng.integers(0, 2**31-1)),
        "charlie": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_bob": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_charlie": int(_rng.integers(0, 2**31-1)),
    }
    for _ in range(5)
]

@pytest.mark.parametrize(
    "psi_b,psi_c,seeds",
    [(*inp, s) for inp, s in itertools.product(_dual_inputs, _dual_seeds)]
)
def test_teledata_dual_randomized(psi_b, psi_c, seeds):
    outs_b, outs_c = dual_trial(psi_b, psi_c, seeds)
    # Ensure each destination receives its corresponding state
    def contains_exact(target, arrs):
        return any(verify_same_state_vector(x, target) for x in arrs)
    assert contains_exact(psi_b, outs_b)
    assert contains_exact(psi_c, outs_c)

