"""
This module implements the quantum manager for ket vector states.
"""
from __future__ import annotations

import math

import numpy as np
from numpy import array

from .base import QuantumManager, QuantumManagerDenseQubit
from ..quantum_state import KetState, OneDimensionInput
from ..quantum_utils import measure_entangled_state_with_cache_ket, measure_multiple_with_cache_ket, measure_state_with_cache_ket
from ...constants import KET_VECTOR_FORMALISM

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...components.circuit import Circuit


# --- Fast gate-contraction helpers -------------------------------------------
# Instead of building the full 2^k x 2^k unitary (and a qutip swap matrix) for an
# entangled group and matmul-ing it against the state -- O(4^k) per gate -- the
# fast path below contracts each small gate directly into the state tensor on
# only the axes it touches (O(2^k) per gate, like a plain statevector simulator).
# No full operator and no swap matrix are ever formed. For any gate it does not
# implement it falls back to the stock matrix path, so it can never silently
# mishandle a circuit.

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Fixed gate matrices, big-endian over the gate's qubit list (first qubit = MSB),
# matching the kron ordering SeQUeNCe uses to build compound states.
_H = np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_CCX = np.eye(8, dtype=complex)
_CCX[[6, 7]] = _CCX[[7, 6]]   # flip target when both controls = 1

# name -> (matrix-or-builder, num_qubits). A builder is callable(arg)->matrix.
_FIXED = {
    "h": (_H, 1), "x": (_X, 1), "y": (_Y, 1), "z": (_Z, 1),
    "s": (_S, 1), "sdg": (_SDG, 1), "t": (_T, 1),
    "cx": (_CX, 2), "cz": (_CZ, 2), "swap": (_SWAP, 2), "ccx": (_CCX, 3),
}


def _phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _gate_matrix(name: str, arg):
    if name in _FIXED:
        return _FIXED[name][0]
    if name == "phase":
        return _phase(arg)
    raise KeyError(name)


_SUPPORTED = set(_FIXED) | {"phase"}


def _apply(tensor: np.ndarray, axes: list[int], gate: np.ndarray) -> np.ndarray:
    """Contract an m-qubit ``gate`` into ``tensor`` on ``axes`` (gate-qubit order).

    ``tensor`` is the statevector viewed as an n-axis array of shape (2,)*n -- one
    axis per qubit. This applies ``gate`` to the ``m`` qubits in ``axes`` without
    ever forming the full 2^n x 2^n operator. Cost is O(2^n), vs O(4^n) for the
    matrix path.

    Worked example -- X on qubit 1 of 3 (axes=[1], gate=2x2):
        tensor[b0][b1][b2]                       # the 8 amplitudes as a 2x2x2 grid
        moveaxis(.., [1], [0]) -> [b1][b0][b2]   # bring the target qubit's axis to the front
        reshape(2, -1) -> a (2, 4) slab          # rows = target qubit (0/1), cols = the others
        gate @ slab                              # one small matmul hits the qubit across all
                                                 #   configs of the other qubits at once
        reshape + moveaxis back                  # restore the original axis order
    """
    m = len(axes)
    front = list(range(m))                       # destination axes 0..m-1
    t = np.moveaxis(tensor, axes, front)         # target qubit(s) -> front (a view; no copy)
    shp = t.shape
    # Collapse the n axes into (2^m gate-qubit rows, everything-else cols), apply the
    # 2^m x 2^m gate, then restore the n-axis shape.
    t = (gate @ t.reshape(2 ** m, -1)).reshape(shp)
    return np.moveaxis(t, front, axes)           # move the axes back where they were


@QuantumManager.register(KET_VECTOR_FORMALISM)
class QuantumManagerKet(QuantumManagerDenseQubit):
    """Class to track and manage quantum states with the ket vector formalism."""

    def __init__(self):
        super().__init__()

    def new(self, state: OneDimensionInput = (complex(1), complex(0))) -> int:
        """Method to create a new ket state.

        Args:
            state (OneDimensionInput): 1D state-vector amplitudes.

        Returns:
            int: the key of the new state.
        """
        key = self._least_available
        self._least_available += 1
        self.states[key] = KetState(state, [key])
        return key

    def run_circuit(self, circuit, keys, meas_samp=None):
        """Fast gate-contraction path. Apply `circuit` to the qubits named by
        `keys`, returning {} or measurement results -- same contract as
        old_run_circuit, but applying each gate by tensor contraction instead of
        building the full matrix. Falls back to old_run_circuit for any
        unsupported gate."""

        # Fallback: if the circuit uses any gate we don't have a fast matrix for,
        # defer to the stock matrix path so we never silently mishandle a circuit.
        if any(g[0] not in _SUPPORTED for g in circuit.gates):
            return self.old_run_circuit(circuit, keys, meas_samp)

        # Run only the input assertions (circuit size matches len(keys); meas_samp
        # present when measuring) -- the same preamble old_run_circuit runs before
        # its matrix path, which is the path we're replacing below.
        self._validate_circuit_run(circuit, keys, meas_samp)

        # --- Assemble the working statevector --------------------------------
        # `keys` may span several separate KetStates (e.g. two memories that aren't
        # entangled yet). Join the distinct ones into one flat vector via kron and
        # record `all_keys` (the qubit order of that vector). This is the same
        # state-combining step _prepare_circuit does -- minus building any operator.
        old_states, all_keys = [], []
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:     # skip keys already pulled in
                old_states.append(qstate.state)
                all_keys += qstate.keys
        state = np.array([1], dtype=complex)
        for s in old_states:
            state = np.kron(state, s)

        # --- Apply each gate by contraction ----------------------------------
        k = len(all_keys)
        tensor = state.reshape((2,) * k)           # flat 2^k vector -> k-axis grid
        key_to_axis = {key: i for i, key in enumerate(all_keys)}
        for name, indices, arg in circuit.gates:
            # `indices` are positions within `keys`; map each to its axis in the
            # assembled tensor (gate-qubit order, e.g. [control, target] for cx).
            axes = [key_to_axis[keys[i]] for i in indices]
            tensor = _apply(tensor, axes, _gate_matrix(name, arg))
        flat = tensor.reshape(-1)                  # back to a flat 2^k vector

        # --- Store result, or measure ----------------------------------------
        if len(circuit.measured_qubits) == 0:
            # No measurement: every key now shares this one (possibly larger) ket.
            new_ket = KetState(flat, all_keys)
            for key in all_keys:
                self.states[key] = new_ket
            return {}

        # Measurement: circuit.measured_qubits are positions in `keys`.
        meas_keys = [keys[i] for i in circuit.measured_qubits]
        return self._measure(flat, meas_keys, all_keys, meas_samp)

    def old_run_circuit(self, circuit: Circuit, keys: list[int], meas_samp=None) -> dict[int, int]:
        """Method to run a circuit on a given list of keys.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys to apply circuit to.
            meas_samp (float): random sample used for measurement result.

        Returns:
            If measurement, dict[int, int]: dictionary mapping qstate keys to measurement results.
            If non-measurement, dict: empty dictionary.
        """
        self._validate_circuit_run(circuit, keys, meas_samp)
        new_state, all_keys, circ_mat = self._prepare_circuit(circuit, keys)

        new_state = circ_mat @ new_state

        if len(circuit.measured_qubits) == 0:
            # set state, return no measurement result
            new_ket = KetState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_ket
            return {}
        else:
            # measure state (state reassignment done in _measure method)
            keys = [all_keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, keys, all_keys, meas_samp)

    def set(self, keys: list[int], amplitudes: OneDimensionInput) -> None:
        """Set the quantum state for the given keys.

        Args:
            keys (list[int]): list of keys of the quantum state.
            amplitudes (OneDimensionInput): amplitudes to set the state to.
        """
        new_state = KetState(amplitudes, keys)
        for key in keys:
            self.states[key] = new_state

    def get_ascending_keys(self, key: int) -> KetState:
        """Method to get quantum state stored at an index.
           Reorders qubits (in-place) in ascending order of keys before returning.

        Args:
            key (int): key for quantum state.

        Returns:
            KetState: quantum state at supplied key.
        """
        state = super().get(key)
        self.reorder_qubits_ascending_keys(state)
        return state

    def reorder_qubits_ascending_keys(self, state: KetState) -> None:
        """Update the quantum state (in-place) to match the ascending order of keys.
           Meanwhile, the reordered state is also set in the quantum manager.
        
        Args:
            state (KetState): The quantum state to reorder.
        """
        target_all_keys = sorted(state.keys)
        if state.keys != target_all_keys:
            _, swap_matrix = self._swap_qubits(state.keys, target_all_keys)
            reordered_state = swap_matrix @ state.state
            state.state = reordered_state
            self.set(target_all_keys, reordered_state.tolist())

    def set_to_zero(self, key: int) -> None:
        """Set the qubit at the given key to the |0> state.

        Args:
            key (int): key of the qubit to set to |0>.
        """
        self.set([key], [complex(1), complex(0)])

    def set_to_one(self, key: int) -> None:
        """Set the qubit at the given key to the |1> state.

        Args:
            key (int): key of the qubit to set to |1>.
        """
        self.set([key], [complex(0), complex(1)])

    def _measure(self, state, keys, all_keys, meas_samp):
        """Single-qubit measurement by tracing out the measured axis (O(2^k))
        instead of building full 2^k projector operators (the stock path's
        O(4^k), which dominates runtime once gate application is contracted).

        Only the single-qubit case is accelerated -- that is every telegate Bell
        measurement and every final-qubit readout. Multi-qubit measurement (rare
        / unused here) falls back to the stock implementation in _old_measure.
        """
        if len(keys) != 1:
            return self._old_measure(state, keys, all_keys, meas_samp)

        key = keys[0]
        k = len(all_keys)
        arr = np.asarray(state, dtype=complex)

        if k == 1:
            prob_0 = float((arr[0].conjugate() * arr[0]).real)
            result = 0 if meas_samp < prob_0 else 1
            new_state = None
        else:
            ax = all_keys.index(key)
            t = np.moveaxis(arr.reshape((2,) * k), ax, 0)   # measured axis to front
            slice0, slice1 = t[0], t[1]                      # (k-1)-qubit branches
            prob_0 = float(np.vdot(slice0, slice0).real)
            if meas_samp < prob_0:
                result = 0
                new_state = (slice0 / np.sqrt(prob_0)).reshape(-1)
            else:
                result = 1
                new_state = (slice1 / np.sqrt(1.0 - prob_0)).reshape(-1)

        all_keys.remove(key)

        # Reassign states exactly as the stock _old_measure does.
        basis = (np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))
        self.states[key] = KetState(basis[result], [key])
        if len(all_keys) > 0:
            new_obj = KetState(new_state, all_keys)
            for rem in all_keys:
                self.states[rem] = new_obj
        return {key: result}

    def _old_measure(self, state: list[complex], keys: list[int], all_keys: list[int], meas_samp: float) -> dict[int, int]:
        """Method to measure qubits at given keys.

        SHOULD NOT be called individually; only from circuit method (unless for unit testing purposes).
        Modifies quantum state of all qubits given by all_keys.

        Args:
            state (list[complex]): state to measure.
            keys (list[int]): list of keys to measure.
            all_keys (list[int]): list of all keys corresponding to state.
            meas_samp (float): random number between 0 and 1 used for measurement.

        Returns:
            dict[int, int]: mapping of measured keys to measurement results.
        """
        if len(keys) == 1:
            if len(all_keys) == 1:
                prob_0 = measure_state_with_cache_ket(tuple(state))
                if meas_samp < prob_0:
                    result = 0
                else:
                    result = 1
            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                state_0, state_1, prob_0 = measure_entangled_state_with_cache_ket(tuple(state), state_index, num_states)
                if meas_samp < prob_0:
                    new_state = array(state_0, dtype=complex)
                    result = 0
                else:
                    new_state = array(state_1, dtype=complex)
                    result = 1

            all_keys.remove(keys[0])

        else:
            # swap states into correct position
            if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
                all_keys, swap_mat = self._swap_qubits(all_keys, keys)
                state = swap_mat @ state

            # calculate meas probabilities and projected states
            len_diff = len(all_keys) - len(keys)
            new_states, probabilities = measure_multiple_with_cache_ket(tuple(state), len(keys), len_diff)

            # choose result, set as new state
            for i in range(int(2 ** len(keys))):
                if meas_samp < sum(probabilities[:i + 1]):
                    result = i
                    new_state = new_states[i]
                    break

            for key in keys:
                all_keys.remove(key)

        result_states = [array([1, 0]), array([0, 1])]
        result_digits = [int(x) for x in bin(result)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        for res, key in zip(result_digits, keys):
            # set to state measured
            new_state_obj = KetState(result_states[res], [key])
            self.states[key] = new_state_obj

        if len(all_keys) > 0:
            new_state_obj = KetState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj

        return dict(zip(keys, result_digits))
