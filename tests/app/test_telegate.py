"""
Telegate Test Suite

This module contains comprehensive tests for the telegate (distributed CNOT) protocol.
It tests both single telegate operations (Alice-Bob) and dual telegate operations
(Alice-Bob-Charlie) with truly random input states and seeds.

Test Categories:
1. Single telegate tests: Test CNOT between Alice (control) and Bob (target)
2. Dual telegate tests: Test concurrent CNOTs between Alice-Bob and Alice-Charlie

All tests use randomly generated quantum states and seeds for comprehensive coverage.
Each test run uses different random states generated from a seeded random number generator.
"""

import os
import math
import itertools
import numpy as np
import pytest
from sequence.topology.dqc_net_topo import DQCNetTopo
from sequence.app.teleportation import TelegateApp
import sequence.utils.log as log
from sequence.kernel.quantum_utils import verify_same_state_vector
from sequence.constants import MILLISECOND

# Configs live next to this test file (tests/app/), so the suite is
# self-contained and runs regardless of the current working directory.
_CFG = os.path.dirname(os.path.abspath(__file__))



# Deterministic random inputs and seeds for reproducible test grids
_rng = np.random.default_rng(2025)


def _random_state(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random normalized quantum state.
    
    Args:
        rng: Random number generator
        
    Returns:
        np.ndarray: Random normalized 2-qubit state vector
    """
    # Random complex 2-vector, normalized
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    vec = vec.astype(complex)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1, 0], dtype=complex)
    return vec / norm


# Generate random states for testing
# Create multiple random states for comprehensive testing
NUM_RANDOM_STATES = 5  # Number of random states to generate

# Generate random states for Alice and Bob
ALICE_PSIS = [_random_state(_rng) for _ in range(NUM_RANDOM_STATES)]
BOB_PSIS = [_random_state(_rng) for _ in range(NUM_RANDOM_STATES)]

# Generate random seeds for reproducible tests
SINGLE_SEEDS = [
    {
        "alice": int(_rng.integers(0, 2**31-1)),
        "bob": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_bob": int(_rng.integers(0, 2**31-1)),
    }
    for _ in range(NUM_RANDOM_STATES)
]


def _all_nodes(topo):
    """
    Extract all nodes from a topology object.
    
    Args:
        topo: Network topology object
        
    Returns:
        list: List of all nodes in the topology
    """
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


def expected_cnot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the expected result of a CNOT operation.
    
    This function computes what the result should be for a CNOT gate
    applied to input states a (control) and b (target).
    
    For CNOT: |control,target⟩ → |control, target ⊕ control⟩
    - |00⟩ → |00⟩ (no change)
    - |01⟩ → |01⟩ (no change)
    - |10⟩ → |11⟩ (target flipped)
    - |11⟩ → |10⟩ (target flipped)
    
    Args:
        a: Control qubit state vector
        b: Target qubit state vector
        
    Returns:
        np.ndarray: Expected 4-element joint state after CNOT
    """
    a0, a1 = a
    b0, b1 = b
    # Standard ordering: |00>, |01>, |10>, |11>
    return np.array([a0*b0,  # |00> → |00>
                     a0*b1,  # |01> → |01>
                     a1*b1,  # |10> → |11> (target flipped)
                     a1*b0], # |11> → |10> (target flipped)
                    dtype=complex)


def single_trial(a_init: np.ndarray, b_init: np.ndarray, seeds=None):
    """
    Execute a single telegate operation between Alice and Bob.
    
    This function sets up a 2-node network, prepares input states,
    executes a telegate CNOT operation, and returns the quantum states
    directly from the quantum state manager.
    
    Args:
        a_init: Alice's initial control qubit state
        b_init: Bob's initial target qubit state
        seeds: Random seeds for reproducible results
        
    Returns:
        tuple: (alice_state, bob_state) - Final quantum states from both nodes
        
    Raises:
        RuntimeError: If telegate operation fails
    """
    # Set up the 2-node network
    topo = DQCNetTopo(os.path.join(_CFG, "teleport_2node.json"))
    tl = topo.tl

    # Log input states and expected CNOT at the beginning
    log.logger.info(f"[telegate_test] === TELEGATE TEST START ===")
    log.logger.info(f"[telegate_test] Input states:")
    log.logger.info(f"[telegate_test]   a_init = {a_init}")
    log.logger.info(f"[telegate_test]   b_init = {b_init}")
    log.logger.info(f"[telegate_test]   seeds = {seeds}")
    
    # Calculate and log expected CNOT result
    expected = expected_cnot(a_init, b_init)
    log.logger.info(f"[telegate_test] Expected CNOT result:")
    for i, val in enumerate(expected):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    nodes = _all_nodes(topo)
    alice = next(n for n in nodes if getattr(n, 'name', '')=="alice")
    bob = next(n for n in nodes if getattr(n, 'name', '')=="bob")
    bsm_nodes = [n for n in nodes if 'BSM' in getattr(n, 'name', '').upper()]
    bsm_ab = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_bob"), None)
    
    # Set seeds
    if "alice" in seeds:
        try: 
            alice.set_seed(seeds["alice"])
        except Exception: 
            pass
    if "bob" in seeds:
        try: 
            bob.set_seed(seeds["bob"])
        except Exception: 
            pass
    if bsm_ab is not None and "BSM_alice_bob" in seeds:
        try: 
            bsm_ab.set_seed(seeds["BSM_alice_bob"])
        except Exception: 
            pass
    
    # Prepare |a> and |b> in Alice's and Bob's data qubits
    a_key = alice.components[alice.data_memo_arr_name].memories[0].qstate_key
    b_key = bob.components[bob.data_memo_arr_name].memories[0].qstate_key
    alice.timeline.quantum_manager.set([a_key], a_init.astype(complex))
    bob.timeline.quantum_manager.set([b_key], b_init.astype(complex))
    
    # Attach the TelegateApp on both nodes
    A = TelegateApp(alice)
    B = TelegateApp(bob)
    
    # Set target slot for Bob (responder) and set current step
    B.set_target_slot_for_step(0, 0)  # step=0, slot=0
    B._current_step = 0  # Set the current step for Bob
    
    # Kick off telegate CNOT
    A.start(responder=bob.name, start_t=10*MILLISECOND, end_t=100*MILLISECOND, 
            memory_size=1, fidelity=0.8, control_src=0, target_src=0, step=0)
    
    # Run the simulation
    tl.init()
    tl.run()
    
    # Wait a bit more to ensure entanglement is established
    tl.run()
    
    # Read out the final quantum states directly from the quantum state manager
    # After a distributed CNOT, Alice's control qubit and Bob's target qubit are entangled
    # Both will have 4-dimensional states representing their part of the joint entangled system
    
    # Read Alice's control qubit state (should be 4-dimensional after entanglement and Z correction)
    alice_control_memory = alice.components[alice.data_memo_arr_name].memories[0]
    alice_control_key = alice_control_memory.qstate_key
    alice_control_state = np.array(alice.timeline.quantum_manager.get(alice_control_key).state, dtype=complex)
    
    # Read Bob's target qubit state (should be 4-dimensional after entanglement and X correction)
    bob_target_memory = bob.components[bob.data_memo_arr_name].memories[0]
    bob_target_key = bob_target_memory.qstate_key
    bob_target_state = np.array(bob.timeline.quantum_manager.get(bob_target_key).state, dtype=complex)
    
    # Log the final results
    log.logger.info(f"[telegate_test] === TELEGATE TEST RESULTS ===")
    log.logger.info(f"[telegate_test] Alice's final state:")
    for i, val in enumerate(alice_control_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Bob's final state:")
    for i, val in enumerate(bob_target_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Log expected CNOT result again for comparison
    log.logger.info(f"[telegate_test] Expected CNOT result:")
    for i, val in enumerate(expected):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Both states should be 4-dimensional after the distributed CNOT
    if len(alice_control_state) == 4 and len(bob_target_state) == 4:
        return alice_control_state, bob_target_state, expected
    
    # If states are not 4-dimensional, raise an error
    raise RuntimeError(f"Telegate failed - Alice state dim: {len(alice_control_state)}, Bob state dim: {len(bob_target_state)}")


@pytest.mark.parametrize(
    "a_init,b_init,seeds",
    list(itertools.product(ALICE_PSIS, BOB_PSIS, SINGLE_SEEDS))
)
def test_telegate_single_randomized(a_init, b_init, seeds):
    """
    Test single telegate operation with truly random inputs.
    
    This test verifies that a distributed CNOT operation between Alice and Bob
    produces the correct result for randomly generated input states by checking 
    quantum states directly from the quantum state manager.
    
    Args:
        a_init: Alice's randomly generated control qubit state
        b_init: Bob's randomly generated target qubit state
        seeds: Random seeds for reproducible results
    """
    alice_state, bob_state, expected = single_trial(a_init, b_init, seeds)
    assert alice_state.shape == (4,) and bob_state.shape == (4,)  # Should be 4-element joint states
    
    print(f"\n=== TELEGATE TEST ===")
    print(f"Input states:")
    print(f"  a_init = {a_init}")
    print(f"  b_init = {b_init}")
    print(f"  seeds = {seeds}")
    print(f"\nAlice's final state:")
    for i, val in enumerate(alice_state):
        print(f"  [{i}] = {val}")
    print(f"\nBob's final state:")
    for i, val in enumerate(bob_state):
        print(f"  [{i}] = {val}")
    print(f"\nExpected result:")
    for i, val in enumerate(expected):
        print(f"  [{i}] = {val}")
    
    # Log the actual results and comparison
    log.logger.info(f"[telegate_test] === TELEGATE TEST RESULTS ===")
    log.logger.info(f"[telegate_test] Alice's final state:")
    for i, val in enumerate(alice_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Bob's final state:")
    for i, val in enumerate(bob_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Log expected CNOT result again for comparison
    log.logger.info(f"[telegate_test] Expected CNOT result:")
    for i, val in enumerate(expected):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Check Alice's state against expected result (with permutation tolerance)
    # Check both [0,1,2,3] and [0,2,1,3] orderings up to global phase
    tolerance = 1e-10
    
    # Check exact match first
    if np.allclose(alice_state, expected, atol=tolerance):
        print(f"\n SUCCESS: Alice's state matches expected result exactly")
        print(f"===============================\n")
        log.logger.info(f"[telegate_test] SUCCESS: Alice's state matches expected result exactly")
        return
    
    # Check global phase tolerance for original order [0,1,2,3]
    phase_factors = []
    for i in range(len(alice_state)):
        if abs(expected[i]) > tolerance:
            phase_factor = alice_state[i] / expected[i]
            phase_factors.append(phase_factor)
    
    if phase_factors:
        phase_factor = phase_factors[0]
        all_same_phase = all(np.isclose(pf, phase_factor, atol=tolerance) for pf in phase_factors[1:])
        if all_same_phase:
            scaled_expected = phase_factor * expected
            if np.allclose(alice_state, scaled_expected, atol=tolerance):
                print(f"\n SUCCESS: Alice's state matches expected result with global phase [0,1,2,3]")
                print(f"===============================\n")
                log.logger.info(f"[telegate_test] SUCCESS: Alice's state matches expected result with global phase [0,1,2,3]")
                return
    
    # Check permutation [0,2,1,3] - swap elements 1 and 2
    permuted_expected = np.array([expected[0], expected[2], expected[1], expected[3]])
    
    # Check exact match with permutation
    if np.allclose(alice_state, permuted_expected, atol=tolerance):
        print(f"\n SUCCESS: Alice's state matches expected result with permutation [0,2,1,3]")
        print(f"===============================\n")
        log.logger.info(f"[telegate_test] SUCCESS: Alice's state matches expected result with permutation [0,2,1,3]")
        return
    
    # Check global phase tolerance for permutation [0,2,1,3]
    phase_factors_perm = []
    for i in range(len(alice_state)):
        if abs(permuted_expected[i]) > tolerance:
            phase_factor = alice_state[i] / permuted_expected[i]
            phase_factors_perm.append(phase_factor)
    
    if phase_factors_perm:
        phase_factor_perm = phase_factors_perm[0]
        all_same_phase_perm = all(np.isclose(pf, phase_factor_perm, atol=tolerance) for pf in phase_factors_perm[1:])
        if all_same_phase_perm:
            scaled_permuted_expected = phase_factor_perm * permuted_expected
            if np.allclose(alice_state, scaled_permuted_expected, atol=tolerance):
                print(f"\n SUCCESS: Alice's state matches expected result with global phase and permutation [0,2,1,3]")
                print(f"===============================\n")
                log.logger.info(f"[telegate_test] SUCCESS: Alice's state matches expected result with global phase and permutation [0,2,1,3]")
                return
    
    # If we get here, no match was found
    print(f"\n FAILED: Alice's state does not match expected result in any ordering")
    print(f"===============================\n")
    log.logger.info(f"[telegate_test] FAILED: Alice's state does not match expected result in any ordering")
    log.logger.info(f"[telegate_test] === TELEGATE TEST END ===")
    assert False


def dual_trial(a_init: np.ndarray, b_init: np.ndarray, seeds=None):
    """
    Execute dual telegate operations between Alice-Bob and Alice-Charlie.
    
    This function sets up a 3-node network, prepares input states on Alice,
    executes concurrent telegate CNOT operations to both Bob and Charlie,
    and returns the quantum states directly from the quantum state manager.
    
    Args:
        a_init: Alice's initial control qubit state for Bob telegate
        b_init: Alice's initial control qubit state for Charlie telegate
        seeds: Random seeds for reproducible results
        
    Returns:
        tuple: (alice_state_bob, alice_state_charlie, expected_bob, expected_charlie) - Final quantum states and expected results
        
    Raises:
        RuntimeError: If telegate operation fails
    """
    # Build 3-node topology (Alice, Bob, Charlie)
    topo = DQCNetTopo(os.path.join(_CFG, "teleport_3node.json"))
    tl = topo.tl

    # Log input states and expected CNOT at the beginning
    log.logger.info(f"[telegate_test] === DUAL TELEGATE TEST START ===")
    log.logger.info(f"[telegate_test] Input states:")
    log.logger.info(f"[telegate_test]   a_init (Bob) = {a_init}")
    log.logger.info(f"[telegate_test]   b_init (Charlie) = {b_init}")
    log.logger.info(f"[telegate_test]   seeds = {seeds}")
    
    # Calculate and log expected CNOT results
    expected_bob = expected_cnot(a_init, np.array([1, 0], dtype=complex))  # Bob's target is |0>
    expected_charlie = expected_cnot(b_init, np.array([1, 0], dtype=complex))  # Charlie's target is |0>
    log.logger.info(f"[telegate_test] Expected CNOT result for Bob:")
    for i, val in enumerate(expected_bob):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Expected CNOT result for Charlie:")
    for i, val in enumerate(expected_charlie):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    nodes = _all_nodes(topo)
    alice = next(n for n in nodes if getattr(n, 'name', '')=="alice")
    bob = next(n for n in nodes if getattr(n, 'name', '')=="bob")
    charlie = next(n for n in nodes if getattr(n, 'name', '')=="charlie")
    bsm_nodes = [n for n in nodes if 'BSM' in getattr(n, 'name', '').upper()]
    bsm_ab = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_bob"), None)
    bsm_ac = next((n for n in bsm_nodes if getattr(n, 'name', '') == "BSM_alice_charlie"), None)
    
    # Set seeds
    if "alice" in seeds:
        try: 
            alice.set_seed(seeds["alice"])
        except Exception: 
            pass
    if "bob" in seeds:
        try: 
            bob.set_seed(seeds["bob"])
        except Exception: 
            pass
    if "charlie" in seeds:
        try: 
            charlie.set_seed(seeds["charlie"])
        except Exception: 
            pass
    if bsm_ab is not None and "BSM_alice_bob" in seeds:
        try: 
            bsm_ab.set_seed(seeds["BSM_alice_bob"])
        except Exception: 
            pass
    if bsm_ac is not None and "BSM_alice_charlie" in seeds:
        try: 
            bsm_ac.set_seed(seeds["BSM_alice_charlie"])
        except Exception: 
            pass
    
    # Prepare two control qubits on Alice at indices 0 and 1
    a_key0 = alice.components[alice.data_memo_arr_name].memories[0].qstate_key
    a_key1 = alice.components[alice.data_memo_arr_name].memories[1].qstate_key
    alice.timeline.quantum_manager.set([a_key0], a_init.astype(complex))
    alice.timeline.quantum_manager.set([a_key1], b_init.astype(complex))
    
    # Attach apps on all participants
    A = TelegateApp(alice)
    B = TelegateApp(bob)
    C = TelegateApp(charlie)
    
    # Set target slots for Bob and Charlie (responders) and set current step
    B.set_target_slot_for_step(0, 0)  # step=0, slot=0
    B._current_step = 0  # Set the current step for Bob
    C.set_target_slot_for_step(0, 0)  # step=0, slot=0
    C._current_step = 0  # Set the current step for Charlie
    
    # Launch two concurrent telegate sessions: Alice to Bob (control 0) and Alice to Charlie (control 1)
    start_t1 = 1 * MILLISECOND
    end_t1 = 200 * MILLISECOND
    start_t2 = 1 * MILLISECOND
    end_t2 = 200 * MILLISECOND
    fidelity = 0.1
    mem_size = 1
    
    A.start(responder=bob.name, start_t=start_t1, end_t=end_t1, memory_size=mem_size, 
            fidelity=fidelity, control_src=0, target_src=0, step=0)
    A.start(responder=charlie.name, start_t=start_t2, end_t=end_t2, memory_size=mem_size, 
            fidelity=fidelity, control_src=1, target_src=0, step=0)
    
    # Run
    tl.init()
    tl.run()
    
    # Wait a bit more to ensure entanglement is established
    tl.run()
    
    # Read out the final quantum states directly from the quantum state manager
    # After distributed CNOTs, Alice's control qubits and Bob/Charlie's target qubits are entangled
    
    # Read Alice's control qubit states (should be 4-dimensional after entanglement and Z corrections)
    alice_control_memory_bob = alice.components[alice.data_memo_arr_name].memories[0]
    alice_control_key_bob = alice_control_memory_bob.qstate_key
    alice_control_state_bob = np.array(alice.timeline.quantum_manager.get(alice_control_key_bob).state, dtype=complex)
    
    alice_control_memory_charlie = alice.components[alice.data_memo_arr_name].memories[1]
    alice_control_key_charlie = alice_control_memory_charlie.qstate_key
    alice_control_state_charlie = np.array(alice.timeline.quantum_manager.get(alice_control_key_charlie).state, dtype=complex)
    
    # Read Bob's and Charlie's target qubit states (should be 4-dimensional after entanglement and X corrections)
    bob_target_memory = bob.components[bob.data_memo_arr_name].memories[0]
    bob_target_key = bob_target_memory.qstate_key
    bob_target_state = np.array(bob.timeline.quantum_manager.get(bob_target_key).state, dtype=complex)
    
    charlie_target_memory = charlie.components[charlie.data_memo_arr_name].memories[0]
    charlie_target_key = charlie_target_memory.qstate_key
    charlie_target_state = np.array(charlie.timeline.quantum_manager.get(charlie_target_key).state, dtype=complex)
    
    # Log the final results
    log.logger.info(f"[telegate_test] === DUAL TELEGATE TEST RESULTS ===")
    log.logger.info(f"[telegate_test] Alice's final state (Bob telegate):")
    for i, val in enumerate(alice_control_state_bob):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Alice's final state (Charlie telegate):")
    for i, val in enumerate(alice_control_state_charlie):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Bob's final state:")
    for i, val in enumerate(bob_target_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Charlie's final state:")
    for i, val in enumerate(charlie_target_state):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Log expected CNOT results again for comparison
    log.logger.info(f"[telegate_test] Expected CNOT result for Bob:")
    for i, val in enumerate(expected_bob):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    log.logger.info(f"[telegate_test] Expected CNOT result for Charlie:")
    for i, val in enumerate(expected_charlie):
        log.logger.info(f"[telegate_test]   [{i}] = {val}")
    
    # Both Alice's states should be 4-dimensional after the distributed CNOTs
    if len(alice_control_state_bob) == 4 and len(alice_control_state_charlie) == 4:
        return alice_control_state_bob, alice_control_state_charlie, expected_bob, expected_charlie
    
    # If states are not 4-dimensional, raise an error
    raise RuntimeError(f"Dual telegate failed - Alice Bob state dim: {len(alice_control_state_bob)}, Alice Charlie state dim: {len(alice_control_state_charlie)}")


# Generate random seeds for dual telegate tests
DUAL_SEEDS = [
    {
        "alice": int(_rng.integers(0, 2**31-1)),
        "bob": int(_rng.integers(0, 2**31-1)),
        "charlie": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_bob": int(_rng.integers(0, 2**31-1)),
        "BSM_alice_charlie": int(_rng.integers(0, 2**31-1)),
    }
    for _ in range(NUM_RANDOM_STATES)
]


@pytest.mark.parametrize(
    "a_init,b_init,seeds",
    list(itertools.product(ALICE_PSIS, BOB_PSIS, DUAL_SEEDS))
)
def test_telegate_dual_randomized(a_init, b_init, seeds):
    """
    Test dual telegate operations with truly random inputs.
    
    This test verifies that concurrent distributed CNOT operations between
    Alice-Bob and Alice-Charlie produce the correct results for randomly 
    generated input states by checking quantum states directly from the 
    quantum state manager.
    
    Args:
        a_init: Alice's randomly generated control qubit state for Bob telegate
        b_init: Alice's randomly generated control qubit state for Charlie telegate
        seeds: Random seeds for reproducible results
    """
    alice_state_bob, alice_state_charlie, expected_bob, expected_charlie = dual_trial(a_init, b_init, seeds)
    assert alice_state_bob.shape == (4,) and alice_state_charlie.shape == (4,)  # Should be 4-element joint states
    
    print(f"\n=== DUAL TELEGATE TEST ===")
    print(f"Input states:")
    print(f"  a_init (Bob) = {a_init}")
    print(f"  b_init (Charlie) = {b_init}")
    print(f"  seeds = {seeds}")
    print(f"\nAlice's final state (Bob telegate):")
    for i, val in enumerate(alice_state_bob):
        print(f"  [{i}] = {val}")
    print(f"\nAlice's final state (Charlie telegate):")
    for i, val in enumerate(alice_state_charlie):
        print(f"  [{i}] = {val}")
    print(f"\nExpected result for Bob:")
    for i, val in enumerate(expected_bob):
            print(f"  [{i}] = {val}")
    print(f"\nExpected result for Charlie:")
    for i, val in enumerate(expected_charlie):
            print(f"  [{i}] = {val}")
    
    # Check Alice's states against expected results (with permutation tolerance)
    # Check both [0,1,2,3] and [0,2,1,3] orderings up to global phase
    tolerance = 1e-10
    
    # Test Bob telegate
    bob_success = False
    
    # Check exact match first
    if np.allclose(alice_state_bob, expected_bob, atol=tolerance):
        print(f"\n SUCCESS: Alice's state (Bob) matches expected result exactly")
        bob_success = True
    else:
        # Check global phase tolerance for original order [0,1,2,3]
        phase_factors = []
        for i in range(len(alice_state_bob)):
            if abs(expected_bob[i]) > tolerance:
                phase_factor = alice_state_bob[i] / expected_bob[i]
                phase_factors.append(phase_factor)
        
        if phase_factors:
            phase_factor = phase_factors[0]
            all_same_phase = all(np.isclose(pf, phase_factor, atol=tolerance) for pf in phase_factors[1:])
            if all_same_phase:
                scaled_expected = phase_factor * expected_bob
                if np.allclose(alice_state_bob, scaled_expected, atol=tolerance):
                    print(f"\n SUCCESS: Alice's state (Bob) matches expected result with global phase [0,1,2,3]")
                    bob_success = True
        
        if not bob_success:
            # Check permutation [0,2,1,3] - swap elements 1 and 2
            permuted_expected = np.array([expected_bob[0], expected_bob[2], expected_bob[1], expected_bob[3]])
            
            # Check exact match with permutation
            if np.allclose(alice_state_bob, permuted_expected, atol=tolerance):
                print(f"\n SUCCESS: Alice's state (Bob) matches expected result with permutation [0,2,1,3]")
                bob_success = True
            else:
                # Check global phase tolerance for permutation [0,2,1,3]
                phase_factors_perm = []
                for i in range(len(alice_state_bob)):
                    if abs(permuted_expected[i]) > tolerance:
                        phase_factor = alice_state_bob[i] / permuted_expected[i]
                        phase_factors_perm.append(phase_factor)
                
                if phase_factors_perm:
                    phase_factor_perm = phase_factors_perm[0]
                    all_same_phase_perm = all(np.isclose(pf, phase_factor_perm, atol=tolerance) for pf in phase_factors_perm[1:])
                    if all_same_phase_perm:
                        scaled_permuted_expected = phase_factor_perm * permuted_expected
                        if np.allclose(alice_state_bob, scaled_permuted_expected, atol=tolerance):
                            print(f"\n SUCCESS: Alice's state (Bob) matches expected result with global phase and permutation [0,2,1,3]")
                            bob_success = True
    
    # Test Charlie telegate
    charlie_success = False
    
    # Check exact match first
    if np.allclose(alice_state_charlie, expected_charlie, atol=tolerance):
        print(f"\n SUCCESS: Alice's state (Charlie) matches expected result exactly")
        charlie_success = True
    else:
        # Check global phase tolerance for original order [0,1,2,3]
        phase_factors = []
        for i in range(len(alice_state_charlie)):
            if abs(expected_charlie[i]) > tolerance:
                phase_factor = alice_state_charlie[i] / expected_charlie[i]
                phase_factors.append(phase_factor)
        
        if phase_factors:
            phase_factor = phase_factors[0]
            all_same_phase = all(np.isclose(pf, phase_factor, atol=tolerance) for pf in phase_factors[1:])
            if all_same_phase:
                scaled_expected = phase_factor * expected_charlie
                if np.allclose(alice_state_charlie, scaled_expected, atol=tolerance):
                    print(f"\n SUCCESS: Alice's state (Charlie) matches expected result with global phase [0,1,2,3]")
                    charlie_success = True
        
        if not charlie_success:
            # Check permutation [0,2,1,3] - swap elements 1 and 2
            permuted_expected = np.array([expected_charlie[0], expected_charlie[2], expected_charlie[1], expected_charlie[3]])
            
            # Check exact match with permutation
            if np.allclose(alice_state_charlie, permuted_expected, atol=tolerance):
                print(f"\n SUCCESS: Alice's state (Charlie) matches expected result with permutation [0,2,1,3]")
                charlie_success = True
            else:
                # Check global phase tolerance for permutation [0,2,1,3]
                phase_factors_perm = []
                for i in range(len(alice_state_charlie)):
                    if abs(permuted_expected[i]) > tolerance:
                        phase_factor = alice_state_charlie[i] / permuted_expected[i]
                        phase_factors_perm.append(phase_factor)
                
                if phase_factors_perm:
                    phase_factor_perm = phase_factors_perm[0]
                    all_same_phase_perm = all(np.isclose(pf, phase_factor_perm, atol=tolerance) for pf in phase_factors_perm[1:])
                    if all_same_phase_perm:
                        scaled_permuted_expected = phase_factor_perm * permuted_expected
                        if np.allclose(alice_state_charlie, scaled_permuted_expected, atol=tolerance):
                            print(f"\n SUCCESS: Alice's state (Charlie) matches expected result with global phase and permutation [0,2,1,3]")
                            charlie_success = True
    
    # Both telegates must succeed
    if bob_success and charlie_success:
        print(f"\n SUCCESS: Both dual telegates succeeded")
        print(f"===============================\n")
        return
    
    # If we get here, at least one telegate failed
    print(f"\n FAILED: Dual telegate test failed")
    if not bob_success:
        print(f"  Bob telegate failed")
    if not charlie_success:
        print(f"  Charlie telegate failed")
    print(f"===============================\n")
    assert False