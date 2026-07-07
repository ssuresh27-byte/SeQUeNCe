"""Teledata protocol and app.

Teledata teleports a data qubit onto a remote node: Alice Bell-measures her data
qubit against her half of the pair and sends the classical bits; Bob applies the
Pauli corrections to his comm memory and then SWAPs the recovered state into his
data memory.

Ported from the DQC ``teledata`` package, re-expressed as a
:class:`TeleportationProtocol` registered under ``TELEDATA`` and grouped with its
:class:`TeledataApp`. Mirrors the teleport protocol/app pair.
"""

from enum import Enum, auto
from typing import List, Optional

from ..request_app import RequestApp
from ...components.circuit import Circuit
from ...kernel.process import Process
from ...kernel.event import Event
from ...message import Message
from ...utils import log
from ...constants import TELEDATA
from ...network_management.reservation import Reservation
from ...resource_management.memory_manager import MemoryInfo
from ...topology.node import DQCNode

from .teleportation_base import TeleportationProtocol


class TeledataMsgType(Enum):
    """Types of classical messages used in teledata."""
    MEASUREMENT_RESULT = auto()  # Alice -> Bob: send (x,z) for corrections
    ACK = auto()                 # Bob -> Alice: acknowledge completion


class TeledataMessage(Message):
    """Classical messages used by TeledataProtocol.

    MEASUREMENT_RESULT:
      - reservation: Reservation associated with this entanglement
      - bob_comm_memory_name: Bob's comm memory to correct
      - x_flip, z_flip: Pauli correction bits
      - bob_data_memory_index: target data slot on Bob's side

    ACK:
      - reservation: same reservation for releasing rules/resources
      - bob_comm_memory_name: for matching protocol instances on Alice side
    """
    def __init__(self, msg_type: TeledataMsgType, **kwargs):
        """Build a teledata classical message.

        Args:
            msg_type (TeledataMsgType): which message variant to build; determines
                the required keyword arguments (see class docstring).
            **kwargs: fields for the chosen message type (see class docstring).
        """
        super().__init__(msg_type, 'teledata_app')

        if msg_type is TeledataMsgType.MEASUREMENT_RESULT:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.x_flip: int = kwargs['x_flip']
            self.z_flip: int = kwargs['z_flip']
            self.bob_data_memory_index: int = kwargs.get('bob_data_memory_index', 0)  # Default to 0 if not provided
            self.string = (f'type={TeledataMsgType.MEASUREMENT_RESULT}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'x_flip={self.x_flip}, z_flip={self.z_flip}, '
                           f'bob_data_memory_index={self.bob_data_memory_index}, '
                           f'reservation={self.reservation}')

        elif msg_type is TeledataMsgType.ACK:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.string = (f'type={TeledataMsgType.ACK}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'reservation={self.reservation}')

        else:
            raise Exception(f"TeledataMessage created unknown type of message: {msg_type}")

    def __str__(self):
        return self.string


@TeleportationProtocol.register(TELEDATA)
class TeledataProtocol(TeleportationProtocol):
    """Core teledata logic, per-session instance with clear Alice/Bob roles."""

    # Alice's Bell-state measurement on (data, comm):
    _bsm_circuit = Circuit(2)
    _bsm_circuit.cx(0, 1)
    _bsm_circuit.h(0)
    _bsm_circuit.measure(0)
    _bsm_circuit.measure(1)

    # Bob's Pauli corrections:
    _z_flip_circuit = Circuit(1); _z_flip_circuit.z(0)
    _x_flip_circuit = Circuit(1); _x_flip_circuit.x(0)

    # SWAP circuit to move teleported state from comm to data memory
    _swap_circuit = Circuit(2)
    _swap_circuit.cx(0, 1)  # comm_qubit -> data_qubit
    _swap_circuit.cx(1, 0)  # data_qubit -> comm_qubit
    _swap_circuit.cx(0, 1)  # comm_qubit -> data_qubit

    def __init__(self, owner: DQCNode, alice: bool = False, data_memory_index: Optional[int] = None,
                 remote_node_name: Optional[str] = None):
        """Initialize a teledata protocol instance for one session.

        Args:
            owner (DQCNode): node this protocol instance is attached to.
            alice (bool): True if this node plays the Alice (initiator) role.
            data_memory_index (Optional[int]): index of Alice's data qubit to teleport.
            remote_node_name (Optional[str]): name of the remote node in this session.
        """
        if alice:
            name = f"{owner.name}.teledata.{remote_node_name}.DataMemoryArray[{data_memory_index}]"
        else:
            name = f"{owner.name}.teledata.{remote_node_name}"
        super().__init__(owner, name, TELEDATA, alice=alice, remote_node_name=remote_node_name)
        self.data_memory_index = data_memory_index
        log.logger.debug(f"{self.name}: initialized (data_memory={data_memory_index})")

    # ───────── Alice side ─────────
    def alice_bell_measurement(self, reservation: Reservation):
        """Perform Bell measurement and send classical bits to Bob.

        Args:
            reservation (Reservation): reservation associated with this entanglement.
        """
        assert self.alice_comm_memory is not None
        comm_key = self.alice_comm_memory.qstate_key

        data_array = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        data_key = data_array[self.data_memory_index].qstate_key  # type: ignore[index]

        log.logger.debug(f"{self.name}: alice_bell_measurement data_key={data_key}, comm_key={comm_key}")
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(TeledataProtocol._bsm_circuit, [data_key, comm_key], rnd)
        z, x = meas[data_key], meas[comm_key]

        msg = TeledataMessage(TeledataMsgType.MEASUREMENT_RESULT, bob_comm_memory_name=self.bob_comm_memory_name,
                              x_flip=x, z_flip=z, bob_data_memory_index=self.data_memory_index, reservation=reservation)
        self.owner.send_message(self.remote_node_name, msg)
        log.logger.info(f"{self.name}: sent measurement results to {self.remote_node_name}: "
                        f"bob_comm_memory={self.bob_comm_memory_name}, x={x}, z={z}, reservation={reservation}")

    # ───────── Bob side ─────────
    def received_message(self, src: str, msg: TeledataMessage):
        """Dispatch an incoming classical message to the right handler.

        Args:
            src (str): name of the node that sent the message.
            msg (TeledataMessage): the received teledata message.
        """
        if msg.msg_type == TeledataMsgType.MEASUREMENT_RESULT:
            self.bob_handle_correction(msg)
        else:
            log.logger.warning(f"{self.name}: received unknown message type {msg.msg_type} from {src}")

    def bob_handle_correction(self, msg: TeledataMessage):
        """Apply corrections on Bob's comm memory, swap to data memory, and notify app when done.

        Args:
            msg (TeledataMessage): MEASUREMENT_RESULT message carrying the Pauli
                correction bits and the target data-memory index.
        """
        assert self.bob_comm_memory is not None
        bob_comm_memory_key = self.bob_comm_memory.qstate_key

        log.logger.debug(f"{self.name}: bob_handle_correction, memory={msg.bob_comm_memory_name}, "
                         f"x_flip={msg.x_flip}, z_flip={msg.z_flip}")

        # Apply Pauli corrections to comm memory
        if msg.x_flip:
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(TeledataProtocol._x_flip_circuit, [bob_comm_memory_key], rnd)
            log.logger.info(f"{self.name}: X-flip applied on memory {msg.bob_comm_memory_name}")
        if msg.z_flip:
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(TeledataProtocol._z_flip_circuit, [bob_comm_memory_key], rnd)
            log.logger.info(f"{self.name}: Z-flip applied on memory {msg.bob_comm_memory_name}")

        # Get Bob's data memory key for the target slot
        data_memory_array = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        # Use modulo to handle cases where the target index exceeds available memory slots
        target_slot = msg.bob_data_memory_index % len(data_memory_array.memories)
        bob_data_memory_key = data_memory_array[target_slot].qstate_key

        # SWAP the teleported state from comm memory to data memory
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TeledataProtocol._swap_circuit, 
                                                        [bob_comm_memory_key, bob_data_memory_key], rnd)
        log.logger.info(f"{self.name}: SWAP applied - teleported state moved from comm to data memory slot {target_slot} "
                        f"(requested {msg.bob_data_memory_index})")

        # Notify app that the teleported state is now in the data memory
        self.owner.teledata_app.teledata_complete(bob_data_memory_key)

    def bob_acknowledge_complete(self, reservation: Reservation):
        """Send an ACK back to Alice so she can release the session's resources.

        Args:
            reservation (Reservation): reservation to release on the Alice side.
        """
        msg = TeledataMessage(TeledataMsgType.ACK, bob_comm_memory_name=self.bob_comm_memory_name,
                              reservation=reservation)
        self.owner.send_message(self.remote_node_name, msg)
        log.logger.debug(f"{self.name}: sent ACK to {self.remote_node_name}")


class TeledataApp(RequestApp):
    """TeledataApp orchestrates teledata using per-session TeledataProtocol instances,
    mirroring the structure used by the teleportation app.
    """

    def __init__(self, node: DQCNode):
        """Initialize the teledata app on a node.

        Args:
            node (DQCNode): node this app is attached to.
        """
        super().__init__(node)
        self.name = "teledata_app"
        node.teledata_app = self

        self.results: List = []
        self.data_keys: List[int] = []

        # Maintain multiple concurrent teledata sessions
        self.teledata_protocols: List[TeledataProtocol] = []

        log.logger.debug(f"[TeledataApp:{node.name}] initialized")

    def start(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float, data_src: int):
        """Start a teledata session (this node acts as Alice).

        Args:
            responder (str): name of the remote node receiving the teleported state.
            start_t (int): start time of the reservation (in ps).
            end_t (int): end time of the reservation (in ps).
            memory_size (int): number of memories to reserve.
            fidelity (float): target fidelity of the entangled pair.
            data_src (int): index of the local data qubit to teleport.
        """
        log.logger.debug(f"[TeledataApp:{self.node.name}] start() → responder={responder}, data_src={data_src}")

        # Reserve and generate EPR pair(s)
        super().start(responder, start_t, end_t, memory_size, fidelity)

        # Create a new protocol instance for Alice
        protocol = TeleportationProtocol.create(owner=self.node, alice=True, data_memory_index=data_src,
                                                remote_node_name=responder, protocol_type=TELEDATA)
        
        self.teledata_protocols.append(protocol)

    def get_reservation_result(self, reservation: Reservation, result: bool):
        """Handle the reservation result from the network manager.

        Args:
            reservation (Reservation): the reservation object.
            result (bool): True if the reservation succeeded, False otherwise.
        """
        super().get_reservation_result(reservation, result)
        log.logger.debug(f"[TeledataApp:{self.node.name}] reservation_result → {result}")

    def get_memory(self, info: MemoryInfo):
        """Handle memory updates and wire comm memories to the right protocol.

        Args:
            info (MemoryInfo): information about the memory state change.
        """
        log.logger.debug(f"{self.name}: get_memory(idx={info.index}, state={info.state})")
        if info.index in self.memo_to_reservation and info.state == "ENTANGLED":
            reservation = self.memo_to_reservation[info.index]

            # Try to match an existing Alice-side protocol
            for protocol in list(self.teledata_protocols):
                this_node = getattr(info.memory.owner, 'name', None)
                remote_node = getattr(info, 'remote_node', reservation.responder)
                if this_node == protocol.owner.name and remote_node == protocol.remote_node_name:
                    # Alice side
                    protocol.set_alice_comm_memory_name(getattr(info.memory, 'name', None))
                    protocol.set_alice_comm_memory(info.memory)
                    protocol.set_bob_comm_memory_name(getattr(info, 'remote_memo', None))
                    # Defer Alice's Bell measurement to a same-time, lower-priority
                    # event so Bob's EntanglementGenerationA._entanglement_succeed()
                    # commits the EPR state first (mirrors TeleportApp.get_memory).
                    # Measuring synchronously here can race that commit.
                    time_now = self.node.timeline.now()
                    process = Process(protocol, 'alice_bell_measurement', [reservation])
                    priority = self.node.timeline.schedule_counter
                    event = Event(time_now, process, priority)
                    self.node.timeline.schedule(event)
                    break
            else:
                # Bob side: create a protocol instance and stash comm memory
                protocol = TeleportationProtocol.create(owner=self.node, alice=False,
                                                        remote_node_name=reservation.initiator, protocol_type=TELEDATA)
                protocol.set_bob_comm_memory_name(getattr(info.memory, 'name', None))
                protocol.set_bob_comm_memory(info.memory)
                protocol.set_alice_comm_memory_name(getattr(info, 'remote_memo', None))
                self.teledata_protocols.append(protocol)

    def received_message(self, src: str, msg: TeledataMessage):
        """Route teledata messages to the matching session protocol.

        Args:
            src (str): name of the node that sent the message.
            msg (TeledataMessage): the received teledata message.
        """
        log.logger.debug(f"{self.name} received_message from {src}: {msg}")

        if msg.msg_type is TeledataMsgType.MEASUREMENT_RESULT:
            # Bob receives measurement result from Alice
            for protocol in list(self.teledata_protocols):
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    protocol.received_message(src, msg)
                    # Send ACK back to Alice and close Bob-side protocol
                    protocol.bob_acknowledge_complete(msg.reservation)
                    self.teledata_protocols.remove(protocol)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teledata protocol for msg from {src}")

        elif msg.msg_type is TeledataMsgType.ACK:
            # Alice receives acknowledgment from Bob → close Alice-side protocol
            for protocol in list(self.teledata_protocols):
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    self.teledata_protocols.remove(protocol)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching teledata protocol for ACK from {src}")

    def teledata_complete(self, data_key: int):
        """Called by TeledataProtocol (on Bob) once corrections are applied.

        Args:
            data_key (int): qstate key of Bob's data memory holding the teleported state.
        """
        full_state = self.node.timeline.quantum_manager.get(data_key).state
        try:
            if hasattr(full_state, "__len__") and len(full_state) > 2:
                half = len(full_state) // 2
                psi = full_state[:half]
            else:
                psi = full_state
        except Exception:
            psi = full_state

        log.logger.info(f"{self.name}: teledata done, state={psi}")
        self.data_keys.append(data_key)
        self.results.append(psi)
