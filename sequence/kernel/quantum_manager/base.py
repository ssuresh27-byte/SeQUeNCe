"""Base classes for quantum state managers.

This module defines the root QuantumManager API used to create, store, retrieve, and update quantum states by manager
key. It also provides module-level circuit helpers (`validate_circuit_run`, `swap_qubits`) shared by the dense qubit
managers (ket vector, density matrix); they live outside the class so the non-dense managers don't inherit them.

Supported manager formalisms include:
    - Ket vector
    - Density matrix
    - Fock density matrix
    - Bell diagonal state
    - Stabilizer state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from threading import Lock
from typing import TYPE_CHECKING, Any
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import gate_sequence_product, Gate

from ...constants import KET_VECTOR_FORMALISM

if TYPE_CHECKING:
    from ...components.circuit import Circuit
    from ..quantum_state import State


class QuantumManager(ABC):
    """Class to track and manage quantum states (abstract).

    All states stored are of a single formalism (by default as a ket vector).

    Class Attributes:
        _registry (dict): mapping of formalism names to manager classes.
        _global_formalism_lock (Lock): lock for managing global formalism.
        _global_formalism (str): global formalism.

    Attributes:
        states (dict[int, State]): mapping of state keys to quantum state objects.
        _least_available (int): tracking the total number of quantum states in the quantum network
    """
    _registry: dict = {}
    _global_formalism_lock = Lock()
    _global_formalism: str = KET_VECTOR_FORMALISM

    def __init__(self):
        self.states: dict[int, State] = {}
        self._least_available: int = 0

    @classmethod
    def set_global_manager_formalism(cls, formalism: str):
        """Set the global manager formalism.

        Args:
            formalism (str): The formalism to set as the global manager formalism.
        """
        with cls._global_formalism_lock:
            if formalism not in cls._registry:
                raise ValueError(f"Quantum manager '{formalism}' is not registered.")
            cls._global_formalism = formalism

    @classmethod
    def get_active_formalism(cls):
        with cls._global_formalism_lock:
            return cls._global_formalism

    @classmethod
    def clear_active_formalism(cls):
        with cls._global_formalism_lock:
            cls._global_formalism = KET_VECTOR_FORMALISM

    @classmethod
    def register(cls, name: str, manager_class=None):
        """Register a quantum manager class.

        Args:
            name (str): The name of the quantum manager.
            manager_class (type, optional): The manager class to register.
        """
        if manager_class is not None:
            cls._registry[name] = manager_class
            return None

        def decorator(manager_cls):
            cls._registry[name] = manager_cls
            return manager_cls

        return decorator

    @classmethod
    def create(cls, *args, **kwargs) -> QuantumManager:
        """Create a new instance of the quantum manager.
        """
        active_formalism = cls.get_active_formalism()
        if active_formalism not in cls._registry:
            raise ValueError(f"Quantum manager '{active_formalism}' is not registered.")

        return cls._registry[active_formalism](*args, **kwargs)

    @abstractmethod
    def new(self, state: Any = None) -> int:
        """Method to create a new quantum state.

        Args:
            state (Any): State payload for the new state. Type depends on subclass.

        Returns:
            int: key for new state generated.
        """
        pass

    def get(self, key: int) -> State:
        """Method to get quantum state stored at an index.

        Args:
            key (int): key for quantum state.

        Returns:
            State: quantum state at supplied key.
        """
        return self.states[key]

    @abstractmethod
    def set(self, keys: list[int], state: Any) -> None:
        """Method to set quantum state at a given key(s).

        Args:
            keys (list[int]): key(s) of state(s) to change.
            state (Any): State payload to assign, type determined by type of subclass.
        """

        pass

    def remove(self, key: int) -> None:
        """Method to remove state stored at key.
        
        Args:
            key (int): The key of the state to remove.
        """
        del self.states[key]

    def set_states(self, states: dict):
        """Set multiple quantum states.

        Args:
            states (dict): A dictionary mapping keys to their corresponding quantum states.
        """
        self.states = states


# --- Dense qubit circuit helpers ---------------------------------------------
# Module-level helpers shared by the ket-vector and density-matrix managers only.
# They live outside QuantumManager so the non-dense managers (Bell diagonal, Fock,
# stabilizer) that also subclass it don't inherit them.

def validate_circuit_run(circuit: Circuit, keys: list[int], meas_samp=None) -> None:
    """Validate common dense-qubit circuit inputs.

    Args:
        circuit (Circuit): quantum circuit to apply.
        keys (list[int]): list of keys for quantum states to apply circuit to.
        meas_samp (float): random sample used for measurement.
    """
    if len(keys) != circuit.size:
        raise ValueError("Mismatch between circuit size and supplied qubits.")
    if circuit.measured_qubits and meas_samp is None:
        raise ValueError("Must specify random sample when measuring qubits.")


def swap_qubits(all_keys: list[int], keys: list[int]) -> tuple[list[int], NDArray]:
    """Swap qubits in the circuit.

    Args:
        all_keys (list[int]): The list of all qubit keys.
        keys (list[int]): The list of qubit keys to swap.

    Returns:
        tuple: updated list of all keys and the swap matrix.
    """
    swap_circuit = QubitCircuit(N=len(all_keys))
    for i, key in enumerate(keys):
        j = all_keys.index(key)
        if j != i:
            gate = Gate("SWAP", targets=[i, j])
            swap_circuit.add_gate(gate)
            all_keys[i], all_keys[j] = all_keys[j], all_keys[i]
    swap_mat = gate_sequence_product(swap_circuit.propagators()).full()
    return all_keys, swap_mat
