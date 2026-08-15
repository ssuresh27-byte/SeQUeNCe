"""
This module implements the quantum manager for density matrix states.
"""
from __future__ import annotations

from .base import QuantumManager, swap_qubits, validate_circuit_run
from ..quantum_state import DensityState, OneDimensionInput, TwoDimensionInput
from ..quantum_utils import (identity, kron, measure_entangled_state_with_cache_density, 
                             measure_multiple_with_cache_density, measure_state_with_cache_density)
from ...constants import DENSITY_MATRIX_FORMALISM

import itertools
import numpy as np
from numpy import array
from typing import TYPE_CHECKING
from ...components.circuit import Circuit


@QuantumManager.register(DENSITY_MATRIX_FORMALISM)
class QuantumManagerDensity(QuantumManager):
    """Class to track and manage states with the density matrix formalism."""

    # Single-qubit Pauli matrices used to build noise-channel Kraus operators.
    _PAULI = {"I": np.eye(2, dtype=complex),
              "X": np.array([[0, 1], [1, 0]], dtype=complex),
              "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
              "Z": np.array([[1, 0], [0, -1]], dtype=complex)}

    # Which non-identity Pauli alphabet each noise_type samples over.
    _NOISE_ALPHABET = {"depolarize": "IXYZ", "dephase": "IZ", "bit_flip": "IX"}

    def __init__(self, seed: int | None = None, **kwargs):
        """Initialize a density-matrix manager, optionally with a per-gate noise model.

        With default arguments the manager is noiseless and behaves exactly as before.
        Noise is applied as deterministic CPTP (Kraus) channels on the density matrix
        after each gate -- unlike the stabilizer manager, which samples Pauli branches.

        Args:
            seed (int | None): seed for the measurement-error RNG (reproducibility).
            **kwargs: noise configuration:
                - noise_type (str): channel applied after each gate, one of
                  "depolarize" (all non-identity Paulis), "dephase" (Z only, T2-like),
                  or "bit_flip" (X only). Default "depolarize".
                - one_qubit_gate_fid (float): avg fidelity of 1-qubit gates. Default 1.0.
                - two_qubit_gate_fid (float): avg fidelity of 2-qubit gates. Default 1.0.
                - measurement_fid (float): probability a reported measurement bit is
                  correct (1 - readout bit-flip probability). Default 1.0.

        The infidelity -> channel-strength conversion matches the stabilizer manager:
        for an average gate fidelity F on a d-dimensional system the total error
        probability is (d+1)/d * (1 - F): 1.5*(1-F) for 1-qubit gates (d=2) and
        1.25*(1-F) for 2-qubit gates (d=4), capped at 1.
        """
        super().__init__()
        self.noise_rng = np.random.default_rng(seed)
        self.noise_type: str = kwargs.get("noise_type", "depolarize")
        if self.noise_type not in self._NOISE_ALPHABET:
            raise ValueError(f"Unknown noise_type '{self.noise_type}'. "
                             f"Use one of {sorted(self._NOISE_ALPHABET)}.")
        self.one_qubit_gate_fid: float = self._validate(kwargs.get("one_qubit_gate_fid", 1.0))
        self.two_qubit_gate_fid: float = self._validate(kwargs.get("two_qubit_gate_fid", 1.0))
        self.measurement_fid: float = self._validate(kwargs.get("measurement_fid", 1.0))
        # accounting
        self.gate_1q_count = 0
        self.gate_2q_count = 0
        self.measurement_count = 0
        self.measurement_error_count = 0

    @staticmethod
    def _validate(value: float) -> float:
        """Return a fidelity/probability after checking it lies in [0, 1].

        Args:
            value (float): fidelity/probability to validate.

        Returns:
            float: the validated value (unchanged).

        Raises:
            ValueError: if value is outside [0, 1].
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("Fidelity must be between 0 and 1, inclusive.")
        return value

    def _noise_enabled(self) -> bool:
        """True if any configured fidelity is below 1 (i.e. noise should be applied).

        Returns:
            bool: True if any gate or measurement fidelity is < 1, else False.
        """
        return (self.one_qubit_gate_fid < 1.0
                or self.two_qubit_gate_fid < 1.0
                or self.measurement_fid < 1.0)

    def new(self, state: OneDimensionInput | TwoDimensionInput = ((complex(1), complex(0)), (complex(0), complex(0)))) -> int:
        """Method to create a new density matrix state.
        
        Args:
            state (OneDimensionInput | TwoDimensionInput): 2D density matrix or 1D pure-state array.

        Returns:
            int: key of the new state.
        """
        key = self._least_available
        self._least_available += 1
        self.states[key] = DensityState(state, [key])
        return key

    def run_circuit(self, circuit: Circuit, keys: list[int], meas_samp=None) -> dict[int, int]:
        """Method to run a circuit on a given list of keys.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys to apply circuit to.
            meas_samp (float): random number between 0 and 1 used for measurement.

        Returns:
            If measurement, dict[int, int]: dictionary mapping qstate keys to measurement results.
            If non-measurement, dict: empty dictionary.
        """
        validate_circuit_run(circuit, keys, meas_samp)

        # Noisy path: apply gates one at a time, inserting a noise channel after each
        # (see _run_circuit_noisy). Only taken when a fidelity < 1 is configured.
        if self._noise_enabled():
            return self._run_circuit_noisy(circuit, keys, meas_samp)

        new_state, all_keys, circ_mat = self._prepare_circuit(circuit, keys)

        new_state = circ_mat @ new_state @ circ_mat.conj().T

        if len(circuit.measured_qubits) == 0:
            # set state, return no measurement result
            new_state_obj = DensityState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj
            return {}
        else:
            # measure state (state reassignment done in _measure method)
            keys = [all_keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, keys, all_keys, meas_samp)

    def _prepare_circuit(self, circuit: Circuit, keys: list[int]) -> tuple[np.ndarray, list[int], np.ndarray]:
        """Prepare state and circuit matrices for dense-qubit execution.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys for quantum states to apply circuit to.

        Returns:
            tuple: tuple containing the new state, all keys, and the circuit matrix.
                   Note: the returned circuit matrix contains any necessary swaps to align qubits of new state
        """
        old_states = []
        all_keys = []

        # go through keys and get all unique qstate objects
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(qstate.state)
                all_keys += qstate.keys

        # construct compound state; order qubits
        new_state = [1]
        for state in old_states:
            new_state = kron(new_state, state)

        # get circuit matrix; expand if necessary
        circ_mat = circuit.get_unitary_matrix()
        if circuit.size < len(all_keys):
            # pad size of circuit matrix if necessary
            diff = len(all_keys) - circuit.size
            circ_mat = kron(circ_mat, identity(2 ** diff))

        # apply any necessary swaps
        if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
            all_keys, swap_mat = swap_qubits(all_keys, keys)
            circ_mat = circ_mat @ swap_mat

        return new_state, all_keys, circ_mat

    # ------------------------------------------------------------------ #
    # Per-gate noise model (deterministic CPTP channels on the density    #
    # matrix). Enabled by passing gate fidelities < 1 to the constructor. #
    # ------------------------------------------------------------------ #
    def _run_circuit_noisy(self, circuit: Circuit, keys: list[int], meas_samp) -> dict[int, int]:
        """Run a circuit gate-by-gate, applying a noise channel after each gate.

        Each gate's ideal unitary is applied to the joint density matrix, then a
        `noise_type` channel of strength derived from the gate fidelity acts on that
        gate's qubits. Measurement reporting errors flip the returned bit with
        probability ``1 - measurement_fid``.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys to apply circuit to.
            meas_samp (float): random number between 0 and 1 used for measurement.

        Returns:
            If measurement, dict[int, int]: dictionary mapping qstate keys to measurement results.
            If non-measurement, dict: empty dictionary.
        """

        rho, all_keys = self._merge_state(keys)
        n = len(all_keys)
        pos = {key: i for i, key in enumerate(all_keys)}

        for gate in circuit.gates:
            name, indices = gate[0], gate[1]
            arg = gate[2] if len(gate) > 2 else None
            mapped = [pos[keys[j]] for j in indices]        # gate qubits -> all_keys positions
            g = Circuit(n)
            g.gates.append([name, mapped, arg])
            gmat = g.get_unitary_matrix()
            rho = gmat @ rho @ gmat.conj().T
            rho = self._apply_gate_noise(rho, n, mapped)

        if len(circuit.measured_qubits) == 0:
            new_state_obj = DensityState(rho, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj
            return {}

        measured_keys = [keys[i] for i in circuit.measured_qubits]
        results = self._measure(rho, measured_keys, all_keys, meas_samp)

        # Measurement reporting error: flip each reported bit with prob (1 - fid).
        if self.measurement_fid < 1.0:
            for mk in list(results):
                self.measurement_count += 1
                if self.noise_rng.random() > self.measurement_fid:
                    results[mk] ^= 1
                    self.measurement_error_count += 1
        return results

    def _merge_state(self, keys: list[int]) -> tuple[np.ndarray, list[int]]:
        """Tensor the distinct density-matrix blocks touched by `keys` into one rho.

        Args:
            keys (list[int]): keys whose (possibly separable) density-matrix blocks
                should be merged into one joint state.

        Returns:
            tuple[np.ndarray, list[int]]: (rho, all_keys) where all_keys is the union
                of every involved state's keys, in block order (mirrors
                _prepare_circuit).
        """
        old_states: list[np.ndarray] = []
        all_keys: list[int] = []
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(np.asarray(qstate.state, dtype=complex))
                all_keys += list(qstate.keys)
        rho = np.array([[1.0 + 0j]])
        for state in old_states:
            rho = np.kron(rho, state)
        return rho, all_keys

    def _pauli_labels(self, k: int, noise_type: str | None = None) -> list[str]:
        """Non-identity Pauli strings on k qubits for the given (or configured) noise_type.

        Args:
            k (int): number of qubits the Pauli strings act on.
            noise_type (str | None): noise alphabet to draw from; if None, uses the
                manager's configured noise_type.

        Returns:
            list[str]: all length-k Pauli strings (over the noise_type's alphabet),
                excluding the all-identity string.

        Raises:
            ValueError: if noise_type is not a recognized alphabet.
        """
        nt = noise_type if noise_type is not None else self.noise_type
        if nt not in self._NOISE_ALPHABET:
            raise ValueError(f"Unknown noise_type '{nt}'. Use one of {sorted(self._NOISE_ALPHABET)}.")
        alphabet = self._NOISE_ALPHABET[nt]
        labels = ("".join(t) for t in itertools.product(alphabet, repeat=k))
        return [s for s in labels if any(c != "I" for c in s)]

    def _embed_pauli(self, label: str, positions: list[int], n: int) -> np.ndarray:
        """Full 2^n operator with `label`'s Paulis on `positions`, identity elsewhere.

        Args:
            label (str): Pauli string (e.g. "XZ") whose characters are placed on
                `positions`, in order.
            positions (list[int]): qubit indices each Pauli character acts on.
            n (int): total number of qubits in the joint state.

        Returns:
            np.ndarray: the 2^n x 2^n Pauli operator.
        """
        ops = [self._PAULI["I"]] * n
        for pauli_char, q in zip(label, positions):
            ops[q] = self._PAULI[pauli_char]
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        return full

    def _apply_gate_noise(self, rho: np.ndarray, n: int, positions: list[int]) -> np.ndarray:
        """Apply the noise channel for a gate acting on `positions` (1 or 2 qubits).

        rho -> (1-p) rho + (p/|S|) * sum_{P in S} P rho P^dagger, where S is the set of
        non-identity Paulis selected by noise_type and p is the fidelity-derived error
        probability: 1.5*(1-fid) for 1-qubit gates, 1.25*(1-fid) for 2-qubit gates.

        Args:
            rho (np.ndarray): joint density matrix to apply the channel to.
            n (int): total number of qubits in `rho`.
            positions (list[int]): qubit indices the gate (and thus its noise) acts on.

        Returns:
            np.ndarray: the density matrix after the noise channel. Returns `rho`
                unchanged if the gate is not 1- or 2-qubit or the error probability
                is <= 0.
        """
        k = len(positions)
        if k == 1:
            self.gate_1q_count += 1
            p = min(1.0, 1.5 * (1.0 - self.one_qubit_gate_fid))
        elif k == 2:
            self.gate_2q_count += 1
            p = min(1.0, 1.25 * (1.0 - self.two_qubit_gate_fid))
        else:
            return rho  # noise only modeled for 1- and 2-qubit gates
        if p <= 0.0:
            return rho
        labels = self._pauli_labels(k)
        if not labels:
            return rho
        out = (1.0 - p) * rho
        weight = p / len(labels)
        for label in labels:
            P = self._embed_pauli(label, positions, n)
            out = out + weight * (P @ rho @ P.conj().T)
        return out

    # ------------------------------------------------------------------ #
    # Key-based noise API: inject a channel at specific qubits (not tied   #
    # to a gate). This is the density-matrix noise primitive a caller can  #
    # use to model, e.g., error after a teleported gate.                   #
    # ------------------------------------------------------------------ #
    def apply_noise(self, keys: list[int], noise_type: str, amount: float,
                    is_infidelity: bool = True) -> None:
        """Apply a noise channel to the qubits at `keys` within their joint state.

        rho -> (1-p) rho + (p/|S|) * sum_{P in S} P rho P^dagger, where S is the set of
        non-identity Paulis selected by `noise_type`. Separable target states are merged
        into one joint density matrix first.

        Args:
            keys: 1 or 2 quantum-manager keys the channel acts on.
            noise_type: "depolarize" (all non-identity Paulis), "dephase" (Z only),
                or "bit_flip" (X only).
            amount: channel strength. If is_infidelity (default), treated as an average
                gate infidelity and converted to p = amount*(d+1)/d (d = 2**len(keys));
                otherwise used directly as the channel probability p.
            is_infidelity: whether `amount` is an average gate infidelity or a raw p.

        Returns:
            None.
        """
        if amount <= 0.0:
            return
        keys = list(keys)
        k = len(keys)
        if k not in (1, 2):
            raise ValueError("apply_noise supports 1 or 2 keys.")
        d = 2 ** k
        p = min(1.0, amount * (d + 1) / d if is_infidelity else amount)
        if p <= 0.0:
            return
        rho, all_keys = self._merge_state(keys)      # merge separable blocks if needed
        self.set(all_keys, rho)
        positions = [all_keys.index(key) for key in keys]
        n = len(all_keys)
        labels = self._pauli_labels(k, noise_type)
        if not labels:
            return
        out = (1.0 - p) * rho
        weight = p / len(labels)
        for label in labels:
            P = self._embed_pauli(label, positions, n)
            out = out + weight * (P @ rho @ P.conj().T)
        self.set(all_keys, out)

    def reduce_to(self, keep_keys: list[int]) -> None:
        """Partial-trace the joint state holding `keep_keys` down to just those keys.

        Detaches keep_keys from any other qubits currently sharing their density matrix
        (e.g. measured comm qubits left entangled after a teleported gate) and
        re-registers the reduced state. No-op if the state already contains only keep_keys.

        Args:
            keep_keys (list[int]): keys to retain; all other qubits sharing their
                joint state are traced out.

        Returns:
            None.
        """
        keep_keys = list(keep_keys)
        rho, all_keys = self._merge_state(keep_keys)
        if len(all_keys) == len(keep_keys):
            self.set(all_keys, rho)
            return
        reduced, kept_order = self._partial_trace(rho, all_keys, keep_keys)
        self.set(kept_order, reduced)

    @staticmethod
    def _partial_trace(rho: np.ndarray, keys: list[int], keep: list[int]) -> tuple[np.ndarray, list[int]]:
        """Partial-trace `rho` (ordered by `keys`) down to `keep`.

        Traces out each qubit whose key is not in `keep` by summing its row==col index
        via einsum.

        Args:
            rho (np.ndarray): joint density matrix, ordered by `keys`.
            keys (list[int]): keys labeling rho's qubits, in order.
            keep (list[int]): subset of `keys` to retain.

        Returns:
            tuple[np.ndarray, list[int]]: (reduced_rho, kept_key_order).
        """
        n = len(keys)
        t = np.asarray(rho, dtype=complex).reshape([2] * n + [2] * n)
        row = [chr(ord('a') + i) for i in range(n)]
        col = [chr(ord('a') + n + i) for i in range(n)]
        for i in range(n):
            if keys[i] not in keep:
                col[i] = row[i]                      # trace this qubit
        out_row = [row[i] for i in range(n) if keys[i] in keep]
        out_col = [col[i] for i in range(n) if keys[i] in keep]
        subscript = ''.join(row) + ''.join(col) + '->' + ''.join(out_row) + ''.join(out_col)
        reduced = np.einsum(subscript, t)
        m = len(keep)
        kept_order = [key for key in keys if key in keep]
        return reduced.reshape(2 ** m, 2 ** m), kept_order

    def reset_error_statistics(self) -> None:
        """Clear the gate/measurement accounting counters.

        Returns:
            None.
        """
        self.gate_1q_count = 0
        self.gate_2q_count = 0
        self.measurement_count = 0
        self.measurement_error_count = 0

    def get_error_statistics(self) -> dict[str, int]:
        """Return per-run gate and measurement-error counts.

        Returns:
            dict[str, int]: counts keyed by "gate_1q_count", "gate_2q_count",
                "measurement_count", and "measurement_error_count".
        """
        return {"gate_1q_count": self.gate_1q_count,
                "gate_2q_count": self.gate_2q_count,
                "measurement_count": self.measurement_count,
                "measurement_error_count": self.measurement_error_count}

    def set(self, keys: list[int], state: OneDimensionInput | TwoDimensionInput) -> None:
        """Method to set the quantum state at the given keys.

        The state argument may be a 1D pure-state vector or a 2D density matrix.

        Args:
            keys (list[int]): list of quantum manager keys to modify.
            state: quantum state to set input keys to.

        Returns:
            None.
        """
        new_state = DensityState(state, keys)
        for key in keys:
            self.states[key] = new_state

    def set_to_zero(self, key: int):
        """Set the qubit at the given key to the |0><0| state.
        
        Args:
            key (int): key of the qubit to set to |0><0|.

        Returns:
            None.
        """
        self.set([key], [[complex(1), complex(0)], [complex(0), complex(0)]])

    def set_to_one(self, key: int):
        """Set the qubit at the given key to the |1><1| state.
        
        Args:
            key (int): key of the qubit to set to |1><1|.

        Returns:
            None.
        """
        self.set([key], [[complex(0), complex(0)], [complex(0), complex(1)]])

    def get_ascending_keys(self, key: int) -> DensityState:
        """Method to get quantum state stored at an index.
           Reorders qubits (in-place) in ascending order of keys before returning.

        Args:
            key (int): key for quantum state.

        Returns:
            DensityState: quantum state at supplied key.
        """
        state = super().get(key)
        self.reorder_qubits_ascending_keys(state)
        return state

    def reorder_qubits_ascending_keys(self, state: DensityState) -> None:
        """Update the quantum state (in-place) to match the ascending order of keys.
           Meanwhile, the reordered state is also set in the quantum manager.
        
        Args:
            state (DensityState): The quantum state to reorder.

        Returns:
            None.
        """
        target_all_keys = sorted(state.keys)
        if state.keys != target_all_keys:
            _, swap_matrix = swap_qubits(state.keys, target_all_keys)
            reordered_state = swap_matrix @ state.state @ swap_matrix.conj().T
            state.state = reordered_state
            self.set(target_all_keys, reordered_state.tolist())

    def _measure(self, state: list[list[complex]], keys: list[int], all_keys: list[int], meas_samp: float) -> dict[int, int]:
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
                prob_0 = measure_state_with_cache_density(tuple(map(tuple, state)))
                if meas_samp < prob_0:
                    result = 0
                    new_state = [[1, 0], [0, 0]]
                else:
                    result = 1
                    new_state = [[0, 0], [0, 1]]

            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                state_0, state_1, prob_0 = measure_entangled_state_with_cache_density(tuple(map(tuple, state)), state_index, num_states)
                if meas_samp < prob_0:
                    new_state = array(state_0, dtype=complex)
                    result = 0
                else:
                    new_state = array(state_1, dtype=complex)
                    result = 1

        else:
            # swap states into correct position
            if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
                all_keys, swap_mat = swap_qubits(all_keys, keys)
                state = swap_mat @ state @ swap_mat.conj().T

            # calculate meas probabilities and projected states
            len_diff = len(all_keys) - len(keys)
            state_to_measure = tuple(map(tuple, state))
            new_states, probabilities = measure_multiple_with_cache_density(state_to_measure, len(keys), len_diff)

            # choose result, set as new state
            for i in range(int(2 ** len(keys))):
                if meas_samp < sum(probabilities[:i + 1]):
                    result = i
                    new_state = new_states[i]
                    break

        result_digits = [int(x) for x in bin(result)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        new_state_obj = DensityState(new_state, all_keys)
        for key in all_keys:
            self.states[key] = new_state_obj

        return dict(zip(keys, result_digits))