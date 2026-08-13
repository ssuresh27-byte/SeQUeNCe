"""
This module implements the quantum manager for ket vector states.
"""
from __future__ import annotations

import math

import numpy as np

from .base import QuantumManager, QuantumManagerDenseQubit
from ..quantum_state import KetState, OneDimensionInput
from ...constants import KET_VECTOR_FORMALISM


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
_ROOT_IZ = _INV_SQRT2 * np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=complex)
_MINUS_ROOT_IZ = _INV_SQRT2 * np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype=complex)
_ROOT_IY = _INV_SQRT2 * np.array([[1, 1], [-1, 1]], dtype=complex)
_MINUS_ROOT_IY = _INV_SQRT2 * np.array([[1, -1], [1, 1]], dtype=complex)

# name -> (matrix-or-builder, num_qubits). A builder is callable(arg)->matrix.
# Names match the gate names Circuit emits (see components/circuit.py).
_FIXED = {
    "h": (_H, 1), "x": (_X, 1), "y": (_Y, 1), "z": (_Z, 1),
    "s": (_S, 1), "sdg": (_SDG, 1), "t": (_T, 1),
    "cx": (_CX, 2), "cz": (_CZ, 2), "swap": (_SWAP, 2), "ccx": (_CCX, 3),
    "root_iZ": (_ROOT_IZ, 1), "minus_root_iZ": (_MINUS_ROOT_IZ, 1),
    "root_iY": (_ROOT_IY, 1), "minus_root_iY": (_MINUS_ROOT_IY, 1),
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
        """Apply a circuit to the qubits named by `keys`.

        Each gate is applied by tensor contraction instead of building the full
        2^n operator: O(2^n) per gate rather than O(4^n). Every gate Circuit can
        emit has a fast matrix in `_FIXED`; an unrecognized gate raises.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): keys of the qubits to apply the circuit to, in
                circuit-qubit order; may span several separate KetStates.
            meas_samp (float): random sample in [0, 1) used for measurement;
                required when the circuit measures any qubit.

        Returns:
            dict[int, int]: mapping of each measured key to its outcome, or an
                empty dict when the circuit performs no measurement.
        """

        # Every gate must have a fast matrix; there is no other path. This mirrors
        # the set of gates Circuit can emit, so it should never trigger in practice.
        unsupported = [g[0] for g in circuit.gates if g[0] not in _SUPPORTED]
        if unsupported:
            raise NotImplementedError(
                f"QuantumManagerKet.run_circuit received unsupported gate(s): {unsupported}")

        # Input assertions: circuit size matches len(keys); meas_samp present when
        # measuring.
        self._validate_circuit_run(circuit, keys, meas_samp)

        # --- Assemble the working statevector --------------------------------
        # `keys` may span several separate KetStates; kron the distinct ones into
        # one flat vector and record `all_keys` (its qubit order). Dedup by state
        # *object*, not by keys[0]: teleport collapses/merges can leave two distinct
        # states sharing a keys[0], and keying on it would skip one -- dropping its
        # qubits from `all_keys` (crashing the gate or orphaning keys on store).
        old_states, all_keys, seen = [], [], set()
        for key in keys:
            qstate = self.states[key]
            if id(qstate) not in seen:             # skip states already pulled in
                seen.add(id(qstate))
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
        """Measure the qubits at `keys` and collapse the shared state accordingly.

        Measurement is done by tracing out the measured axes (O(2^k)) rather than
        building full 2^k projector operators (the stock path's O(4^k), which
        dominates runtime once gate application is contracted). This handles any
        number of measured qubits `m`: the single-qubit telegate/readout case is
        just `m == 1`, and the two-qubit teleportation Bell measurement is `m == 2`.

        The measured axes are moved to the front and flattened into one 2^m outcome
        index (with `keys[0]` as the most-significant bit); each outcome's Born
        probability is the squared norm of its branch over the remaining qubits.
        After sampling an outcome, each measured key is reassigned to its basis
        state and the remaining qubits (if any) to the renormalized branch.

        Args:
            state (list[complex]): flat amplitudes of the combined state over
                `all_keys` to measure.
            keys (list[int]): keys to measure, in outcome bit order (keys[0] is the
                most-significant bit).
            all_keys (list[int]): qubit order of `state`.
            meas_samp (float): random sample in [0, 1) selecting the outcome.

        Returns:
            dict[int, int]: mapping of each measured key to its outcome (0 or 1).
        """
        m = len(keys)
        k = len(all_keys)
        arr = np.asarray(state, dtype=complex).reshape((2,) * k)

        # Move the measured axes to the front (in `keys` order) and flatten them
        # into a single 2^m outcome index; the remaining qubits become the columns.
        # Row r then enumerates joint outcomes with keys[0] as the most-significant
        # bit -- the manager's outcome convention.
        meas_axes = [all_keys.index(key) for key in keys]
        t = np.moveaxis(arr, meas_axes, range(m)).reshape(2 ** m, -1)

        # Born-rule probability per outcome (squared norm of each branch), then
        # sample the cumulative distribution the same way the stock path does.
        probs = np.einsum("ij,ij->i", t.conjugate(), t).real
        result = int(np.searchsorted(np.cumsum(probs), meas_samp, side="right"))
        result = min(result, 2 ** m - 1)                 # guard float round-off near 1.0
        bits = [(result >> (m - 1 - j)) & 1 for j in range(m)]

        # Reassign each measured key to its basis state, and the remaining qubits
        # (if any) to the renormalized post-measurement branch.
        basis = (np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))
        for key, bit in zip(keys, bits):
            self.states[key] = KetState(basis[bit], [key])
        rem_keys = [key for key in all_keys if key not in keys]
        if rem_keys:
            new_state = (t[result] / np.sqrt(probs[result])).reshape(-1)
            new_obj = KetState(new_state, rem_keys)
            for rem in rem_keys:
                self.states[rem] = new_obj
        return dict(zip(keys, bits))
