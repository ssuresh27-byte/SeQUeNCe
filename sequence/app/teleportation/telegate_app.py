"""Telegate (remote/distributed CNOT) protocol and app.

Implements an entanglement-assisted distributed CNOT across two nodes:

1. Alice: CNOT(control -> comm_A), measure(comm_A) -> a_bit, send a_bit to Bob.
2. Bob: CNOT(comm_B -> target), H(comm_B), measure(comm_B) -> b_bit, apply X^a.
3. Bob: send ACK(a_bit, b_bit) to Alice.
4. Alice: apply Z^b correction to the control qubit, complete.

This matches the standard distributed CNOT with a Z correction on the control
side (Alice) and an X correction on the target side (Bob).

Ported from the DQC ``telegate`` package, re-expressed as a
:class:`TeleportationProtocol` registered under ``TELEGATE`` and grouped with its
:class:`TelegateApp`. Mirrors the teleport protocol/app pair.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import List, Optional, Callable, Dict

import numpy as np

from ..request_app import RequestApp
from ...components.circuit import Circuit
from ...message import Message
from ...utils import log
from ...constants import TELEGATE
from ...network_management.reservation import Reservation
from ...topology.node import DQCNode
from ...resource_management.memory_manager import MemoryInfo
from ...kernel.process import Process
from ...kernel.event import Event

from .teleportation_base import TeleportationProtocol


class TelegateMsgType(Enum):
    """Message types for telegate protocol communication.

    Attributes:
        A_MEAS_RESULT: Alice to Bob - sends measurement result 'a' from comm qubit
        ACK: Bob to Alice - acknowledges completion with both measurement results
    """
    A_MEAS_RESULT = auto()  # Alice to Bob: send bit 'a' (result of measuring comm_A)
    ACK = auto()  # Bob to Alice: acknowledge completion


class TelegateMessage(Message):
    """Message payloads for telegate protocol communication.

    A_MEAS_RESULT message contains:
        - reservation: Network reservation for the telegate operation
        - bob_comm_memory_name: Name of Bob's communication memory
        - a_bit: Alice's measurement of her comm qubit (0/1)
        - target_memory_index: Index of Bob's target memory

    ACK message contains:
        - reservation: Network reservation for the telegate operation
        - bob_comm_memory_name: Name of Bob's communication memory
        - target_key: Bob's target qubit key
        - a_bit: Alice's measurement result
        - b_bit: Bob's measurement result
    """

    def __init__(self, msg_type: TelegateMsgType, **kwargs):
        super().__init__(msg_type, 'telegate_app')

        if msg_type is TelegateMsgType.A_MEAS_RESULT:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.a_bit: int = kwargs['a_bit']
            self.target_memory_index: int = kwargs.get('target_memory_index', 0)
            self.string = (f'type={TelegateMsgType.A_MEAS_RESULT}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'a_bit={self.a_bit}, target_idx={self.target_memory_index}, '
                           f'reservation={self.reservation}')
        elif msg_type is TelegateMsgType.ACK:
            self.reservation: Reservation = kwargs['reservation']
            self.bob_comm_memory_name: str = kwargs['bob_comm_memory_name']
            self.target_key: Optional[int] = kwargs.get('target_key', None)
            self.a_bit: Optional[int] = kwargs.get('a_bit', None)
            self.b_bit: Optional[int] = kwargs.get('b_bit', None)
            self.string = (f'type={TelegateMsgType.ACK}, '
                           f'bob_comm_memory={self.bob_comm_memory_name}, '
                           f'target_key={self.target_key}, a_bit={self.a_bit}, '
                           f'b_bit={self.b_bit}, reservation={self.reservation}')
        else:
            raise Exception(f"TelegateMessage created unknown type of message: {msg_type}")

    def __str__(self):
        return self.string


@TeleportationProtocol.register(TELEGATE)
class TelegateProtocol(TeleportationProtocol):
    """Per-session protocol for remote CNOT operations.

    Manages a single telegate operation between Alice (control qubit owner) and
    Bob (target qubit owner): quantum operations, measurements, corrections, and
    completion signaling.

    Attributes:
        owner: The DQCNode that owns this protocol instance
        alice: True if this is Alice's protocol instance, False for Bob's
        remote_node_name: Name of the remote node in this telegate operation
        control_memory_index: Index of Alice's control qubit memory
        target_memory_index: Index of Bob's target qubit memory
        alice_comm_memory_name / alice_comm_memory: Alice's communication memory
        bob_comm_memory_name / bob_comm_memory: Bob's communication memory
    """

    # Alice: CNOT (data_control → comm_A) + measure(comm_A)
    _cnot = Circuit(2)
    _cnot.cx(0, 1)

    _hadamard = Circuit(1)
    _hadamard.h(0)

    # Bob: Measure comm qubit
    _z_measure = Circuit(1)
    _z_measure.measure(0)

    # Alice: Z correction on control qubit (if b_bit == 1)
    _z = Circuit(1)
    _z.z(0)

    # Bob: X correction on target qubit (if a_bit == 1)
    _x = Circuit(1)
    _x.x(0)

    def __init__(self, owner: DQCNode, alice: bool = False, control_memory_index: Optional[int] = None,
                 target_memory_index: Optional[int] = None, remote_node_name: Optional[str] = None):
        """Initialize a telegate protocol instance.

        Args:
            owner: The DQCNode that owns this protocol instance
            alice: True if this is Alice's protocol instance, False for Bob's
            control_memory_index: Index of Alice's control qubit memory
            target_memory_index: Index of Bob's target qubit memory
            remote_node_name: Name of the remote node in this telegate operation
        """
        if alice:
            name = (f"{owner.name}.telegate.{remote_node_name}."
                    f"ControlData[{control_memory_index}]toTargetData[{target_memory_index}]")
        else:
            name = f"{owner.name}.telegate.{remote_node_name}"
        super().__init__(owner, name, TELEGATE, alice=alice, remote_node_name=remote_node_name)

        # Data indices are meaningful on their respective sides
        self.control_memory_index = control_memory_index  # Alice side index
        self.target_memory_index = target_memory_index  # Bob side index

    def alice_stage(self, reservation: Reservation):
        """Execute Alice's stage of the telegate protocol.

        Alice performs:
        1. CNOT(control → comm_A)
        2. Measure comm_A to get a_bit
        3. Send a_bit to Bob

        Args:
            reservation: Network reservation for the telegate operation
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_stage called")
        assert self.alice_comm_memory is not None, "Alice comm memory not set"
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        data_key = data_arr[self.control_memory_index].qstate_key  # type: ignore[index]
        comm_key = self.alice_comm_memory.qstate_key

        # Step 1: CNOT(control → comm)
        control_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        comm_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State before CNOT - control qubit: {control_state_before}")
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State before CNOT - comm qubit: {comm_state_before}")

        # 1. Apply CNOT without measurement first to log intermediate state
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying CNOT gate: control qubit (key={data_key}) → communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._cnot, [data_key, comm_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] CNOT gate applied successfully")

        # Log quantum state after CNOT but before measurement
        control_state_after_cnot = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        comm_state_after_cnot = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT (before measurement) - control qubit: {control_state_after_cnot}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT (before measurement) - comm qubit: {comm_state_after_cnot}")

        # 2. Now apply measurement
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z_measure, [comm_key], rnd)
        a_bit = meas[comm_key]
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Communication qubit measurement result: a_bit={a_bit}")

        # Log quantum state after measurement
        control_state_after_measure = self.pretty_ket(self.owner.timeline.quantum_manager.get(data_key).state)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] State after measurement - control qubit: {control_state_after_measure}")

        # 3. Send measurement result to Bob
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Sending A_MEAS_RESULT to {self.remote_node_name}")
        msg = TelegateMessage(TelegateMsgType.A_MEAS_RESULT, bob_comm_memory_name=self.bob_comm_memory_name,
                              a_bit=a_bit, target_memory_index=self.target_memory_index, reservation=reservation)
        self.owner.send_message(self.remote_node_name, msg)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] A_MEAS_RESULT sent successfully")

    def received_message(self, src: str, msg: TelegateMessage):
        """Handle incoming messages for Bob's side of the protocol.

        Args:
            src: Source node name
            msg: Received telegate message
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] received_message from {src}, msg_type={msg.msg_type}")
        if msg.msg_type is TelegateMsgType.A_MEAS_RESULT:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Processing A_MEAS_RESULT from {src}")
            self._bob_stage_with_a(msg)
        else:
            log.logger.warning(f"{self.name}: received unknown message type {msg.type} from {src}")

    def _bob_stage_with_a(self, msg: TelegateMessage):
        """Execute Bob's stage of the telegate protocol.

        Bob performs:
        1. CNOT(comm_B → target)
        2. H(comm_B)
        3. Measure comm_B to get b_bit
        4. Apply X^a correction to target qubit
        5. Send ACK(a_bit, b_bit) to Alice
        6. Complete protocol

        Args:
            msg: Alice's measurement result message
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] _bob_stage_with_a called")
        assert self.bob_comm_memory is not None, "Bob comm memory not set"

        # Fetch target data key (on Bob side)
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        t_idx = msg.target_memory_index if hasattr(msg, 'target_memory_index') else (self.target_memory_index if self.target_memory_index is not None else 0)
        target_key = data_arr[t_idx].qstate_key  # type: ignore[index]
        comm_key = self.bob_comm_memory.qstate_key

        comm_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_before = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before CNOT - comm qubit: {comm_state_before}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before CNOT - target qubit: {target_state_before}")

        # 1. Apply CNOT
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying CNOT gate: comm qubit (key={comm_key}) → target qubit (key={target_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._cnot, [comm_key, target_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] CNOT gate applied successfully")

        # Log quantum state after CNOT
        comm_state_after = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_after = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT - comm qubit: {comm_state_after}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after CNOT - target qubit: {target_state_after}")

        # 2. Apply Hadamard gate on comm qubit
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying Hadamard gate to communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._hadamard, [comm_key], rnd)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Hadamard gate applied successfully")

        comm_state_after_h = self.pretty_ket(self.owner.timeline.quantum_manager.get(comm_key).state)
        target_state_after_h = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after Hadamard - comm qubit: {comm_state_after_h}")
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] State after Hadamard - target qubit: {target_state_after_h}")

        # 3. Measure comm qubit
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Measuring communication qubit (key={comm_key})")
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z_measure, [comm_key], rnd)
        b_bit = meas[comm_key]
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Communication qubit measurement result: b_bit={b_bit}")

        # Step 4: Apply X^a correction (uses Alice's bit)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Alice's measurement result: a_bit={msg.a_bit}")
        if msg.a_bit:
            target_state_before_x = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before X correction - target qubit: {target_state_before_x}")

            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying X correction to target qubit (key={target_key}) based on Alice's measurement")
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._x, [target_key], rnd)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] X correction applied successfully")

            target_state_after_x = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] State after X correction - target qubit: {target_state_after_x}")
        else:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] No X correction needed (Alice's measurement was 0)")
            target_state_no_x = self.pretty_ket(self.owner.timeline.quantum_manager.get(target_key).state)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] State after measurement (no X correction) - target qubit: {target_state_no_x}")

        # Send ACK to Alice (with both bits)
        log.logger.warning(f"[telegate_protocol:{self.owner.name}] About to send ACK: a_bit={msg.a_bit}, b_bit={b_bit}, target_key={target_key}")

        self.bob_acknowledge_complete(msg.reservation, a_bit=msg.a_bit, b_bit=b_bit, target_key=target_key)

        # Inform the app that the distributed CNOT has been effected
        role = "alice" if self.alice else "bob"
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Calling telegate_complete with target_key={target_key}, role={role}")
        self.owner.telegate_app.telegate_complete(target_key, role)

        # Remove this protocol from the list after completion
        if self in self.owner.telegate_app.telegate_protocols:
            self.owner.telegate_app.telegate_protocols.remove(self)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Protocol removed from list after completion")

        log.logger.info(f"[telegate_protocol:{self.owner.name}] _bob_stage_with_a completed successfully")

    def alice_z_correction(self, b_bit: int):
        """Apply Alice's Z correction based on Bob's measurement result.

        Called when Alice receives the ACK from Bob: applies the Z^b correction
        to Alice's control qubit and completes the protocol.

        Args:
            b_bit: Bob's measurement result (0 or 1)
        """
        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_z_correction called with b_bit={b_bit}")

        # Get the control qubit key
        data_arr = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        ctrl_key = data_arr[self.control_memory_index].qstate_key

        if b_bit:
            control_state_before_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] State before Z correction - control qubit: {control_state_before_z}")

            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Applying Z gate to control qubit (key={ctrl_key}) based on Bob's measurement")
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(TelegateProtocol._z, [ctrl_key], rnd)
            log.logger.info(f"[telegate_protocol:{self.owner.name}] Z gate applied successfully")

            control_state_after_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.warning(f"[telegate_protocol:{self.owner.name}] State after Z correction - control qubit: {control_state_after_z}")
        else:
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] No Z correction needed (Bob's measurement was 0)")
            control_state_no_z = self.pretty_ket(self.owner.timeline.quantum_manager.get(ctrl_key).state)
            log.logger.warning(f"[telegate_protocol:{self.owner.name}] State after ACK (no Z correction) - control qubit: {control_state_no_z}")

        # Complete as alice (initiator/control owner)
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] Calling telegate_complete with ctrl_key={ctrl_key}, role=alice")
        self.owner.telegate_app.telegate_complete(ctrl_key, role="alice")

        # Remove this protocol from the list after completion
        if self in self.owner.telegate_app.telegate_protocols:
            self.owner.telegate_app.telegate_protocols.remove(self)
            log.logger.debug(f"[telegate_protocol:{self.owner.name}] Protocol removed from list after completion")

        log.logger.info(f"[telegate_protocol:{self.owner.name}] alice_z_correction completed successfully")

    def bob_acknowledge_complete(self, reservation: Reservation, a_bit: int, b_bit: int, target_key: int):
        """Send acknowledgment message to Alice with measurement results.

        Args:
            reservation: Network reservation for the telegate operation
            a_bit: Alice's measurement result
            b_bit: Bob's measurement result
            target_key: Bob's target qubit key
        """
        log.logger.debug(f"[telegate_protocol:{self.owner.name}] bob_acknowledge_complete called: a_bit={a_bit}, b_bit={b_bit}, target_key={target_key}")

        msg = TelegateMessage(TelegateMsgType.ACK, bob_comm_memory_name=self.bob_comm_memory_name, reservation=reservation,
                              a_bit=a_bit, b_bit=b_bit, target_key=target_key)
        log.logger.info(f"[telegate_protocol:{self.owner.name}] Sending ACK to {self.remote_node_name}")
        self.owner.send_message(self.remote_node_name, msg)

    def pretty_ket(self, vec, precision=4, tol=1e-10):
        """Convert a state vector into a pretty-printed ket string."""
        # These dumps are only ever fed to INFO/DEBUG log lines, but the calls are
        # evaluated eagerly (as f-string args), so without this guard they format
        # the whole 2^k-amplitude state on every telegate even at WARNING level.
        if not log.logger.isEnabledFor(logging.INFO):
            return "<state dump suppressed; enable INFO logging>"
        v = np.asarray(vec, dtype=complex).flatten()
        terms = []
        for k, a in enumerate(v):
            if abs(a) < tol:
                continue
            re = round(a.real, precision)
            im = round(a.imag, precision)
            # avoid "-0.0"
            if abs(re) < 10**(-precision): re = 0.0
            if abs(im) < 10**(-precision): im = 0.0

            if im == 0:
                coef = f"{re:.{precision}f}"
            elif re == 0:
                coef = f"{im:.{precision}f}i"
            else:
                sign = "+" if im > 0 else "-"
                coef = f"{re:.{precision}f} {sign} {abs(im):.{precision}f}i"
            terms.append(f"({coef}) |{k}⟩")
        return " + ".join(terms) if terms else "0"


class TelegateApp(RequestApp):
    """Per-node teleported-CNOT application.

    Manages telegate operations on a single DQC node, handling both Alice
    (initiator/control owner) and Bob (responder/target owner) roles. Coordinates
    with TelegateProtocol instances to execute distributed CNOT gates.

    Attributes:
        node: The DQCNode that owns this telegate app
        name: Application name ("telegate_app")
        results: List of telegate operation results for debugging/testing
        data_keys: List of data qubit keys from completed operations
        telegate_protocols: List of active TelegateProtocol instances
        _complete_cbs: List of completion callback functions
        _target_slot_by_step: Mapping of step numbers to target memory slots
        _current_step: Current step number for telegate operations
    """

    def __init__(self, node: DQCNode):
        """Initialize a telegate application.

        Args:
            node: The DQCNode that owns this telegate app
        """
        super().__init__(node)
        self.name = "telegate_app"
        node.telegate_app = self

        # Results/keys for debugging or tests
        self.results: List = []
        self.data_keys: List[int] = []

        # Multiple concurrent telegate sessions
        self.telegate_protocols: List[TelegateProtocol] = []

        # Completion callbacks: cb(role:str, data_key:int, step:Optional[int])
        self._complete_cbs: List[Callable] = []

        # Optional: target-owner can predeclare the exact target data slot per step
        # so responder picks the same local slot DQCApp locked
        self._target_slot_by_step: Dict[int, int] = {}
        self._current_step: Optional[int] = None

    def remove_memo_reservation_map(self, index: int) -> None:
        """Idempotent override of the base-class removal.

        ``_early_expire`` clears a reservation's map entry as soon as the gate
        completes, but the base class also schedules a removal at the
        reservation's ``end_time`` (``RequestApp.schedule_reservation``). The
        stock implementation does ``pop(index)``, which would ``KeyError`` on
        that second, now-redundant call. Popping with a default makes the late
        event a harmless no-op.
        """
        self.memo_to_reservation.pop(index, None)

    def on_complete(self, cb: Callable):
        """Register completion callback.

        Args:
            cb: Callback function with signature cb(role, data_key, step)
        """
        self._complete_cbs.append(cb)

    def set_target_slot_for_step(self, step: int, slot: int):
        """Set the target memory slot for a specific step.

        Optional helper for the target owner (responder) to force its local slot
        to match what DQCApp locked.

        Args:
            step: Step number
            slot: Memory slot index
        """
        self._target_slot_by_step[step] = int(slot)

    def set_current_step(self, step: int):
        """Set the current step for the telegate operation.

        Args:
            step: Current step number
        """
        self._current_step = int(step)

    def start(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float,
              control_src: int, target_src: int, step: Optional[int] = None):
        """Start a telegate session from the control owner (initiator).

        Args:
            responder: Name of the responder node (target qubit owner)
            start_t: Start time for entanglement reservation
            end_t: End time for entanglement reservation
            memory_size: Size of memory for entanglement
            fidelity: Required fidelity for entanglement
            control_src: Index of control qubit memory
            target_src: Index of target qubit memory
            step: Optional step number for the operation
        """
        # Reserve entanglement window
        super().start(responder, start_t, end_t, memory_size, fidelity)

        # Create Alice-side protocol on the initiator (control owner)
        log.logger.info(f"[telegate:{self.node.name}] Creating Alice protocol with remote_node_name={responder}")
        protocol = TeleportationProtocol.create(owner=self.node, alice=True, control_memory_index=control_src,
                                                target_memory_index=target_src, remote_node_name=responder, protocol_type=TELEGATE)
        if step is not None:
            protocol._step = step
        self.telegate_protocols.append(protocol)

        # Optionally mirror step for DQCApp callback convenience
        if step is not None:
            self._current_step = step  # DQCApp also sets this

    def get_reservation_result(self, reservation, result: bool):
        """Handle reservation result from the network.

        Args:
            reservation: Network reservation object
            result: Whether the reservation was successful
        """
        super().get_reservation_result(reservation, result)

    def get_memory(self, info):
        """Handle memory entanglement events.

        Called when a memory becomes entangled. Determines whether this node is
        the initiator (Alice) or responder (Bob) and executes the appropriate
        protocol stage.

        Args:
            info: Memory information object containing entanglement details
        """
        log.logger.info(f"[telegate:{getattr(info.memory.owner, 'name', 'unknown')}] get_memory called: index={info.index}, state={info.state}, memo_to_reservation={list(self.memo_to_reservation.keys())}")

        if info.index not in self.memo_to_reservation or info.state != "ENTANGLED":
            log.logger.warning(f"[telegate:{getattr(info.memory.owner, 'name', 'unknown')}] get_memory returning early: index in reservation={info.index in self.memo_to_reservation}, state is ENTANGLED={info.state == 'ENTANGLED'}")
            return

        reservation = self.memo_to_reservation[info.index]
        identity = getattr(reservation, "identity", None)
        this_node = getattr(info.memory.owner, "name", None)

        # The reservation initiator owns the CNOT control; the responder owns the target.
        control_node = reservation.initiator
        target_node = reservation.responder

        log.logger.info(f"[telegate:{this_node}] get_memory: this_node={this_node}, control={control_node}, target={target_node}")

        # Control side → Alice-role (CNOT control → comm, measure, Z-correct).
        if this_node == control_node:
            log.logger.info(f"[telegate:{this_node}] Looking for Alice protocol for target={target_node}")
            for protocol in list(self.telegate_protocols):
                if (protocol.alice and protocol.owner.name == control_node and protocol.remote_node_name == target_node):
                    log.logger.info(f"[telegate:{this_node}] Found Alice protocol, calling alice_stage")
                    protocol.set_alice_comm_memory_name(getattr(info.memory, "name", None))
                    protocol.set_alice_comm_memory(info.memory)
                    protocol.set_bob_comm_memory_name(getattr(info, "remote_memo", None))
                    process = Process(protocol, "alice_stage", [reservation])
                    event = Event(self.node.timeline.now(), process)
                    self.node.timeline.schedule(event)
                    return

            log.logger.error(f"[TelegateApp:{this_node}] no Alice protocol found for target={target_node}, id={identity}")
            raise RuntimeError("Alice-side protocol missing at control owner")

        # Target side → Bob-role (CNOT comm → target, H, measure, X-correct).
        if this_node == target_node:
            step = getattr(self, "_current_step", None)
            if not isinstance(step, int) or step not in self._target_slot_by_step:
                log.logger.error(f"[TelegateApp:{this_node}] missing target slot for step={step}; did DQCApp.set_target_slot_for_step(step, slot) run on TARGET?")
                raise RuntimeError("Responder target slot not specified")

            target_slot = self._target_slot_by_step[step]
            log.logger.info(f"[telegate:{self.node.name}] Creating Bob protocol with remote_node_name={control_node}")
            protocol = TeleportationProtocol.create(owner=self.node, alice=False, control_memory_index=None,
                                                    target_memory_index=target_slot, remote_node_name=control_node, protocol_type=TELEGATE)
            protocol.set_bob_comm_memory_name(getattr(info.memory, "name", None))
            protocol.set_bob_comm_memory(info.memory)
            protocol.set_alice_comm_memory_name(getattr(info, "remote_memo", None))
            self.telegate_protocols.append(protocol)

            return

        # Otherwise ignore (only two parties per telegate)

    def received_message(self, src: str, msg: TelegateMessage):
        """Handle incoming telegate messages.

        Routes messages to the appropriate protocol instance based on message
        type and protocol matching criteria.

        Args:
            src: Source node name
            msg: Received telegate message
        """
        if msg.msg_type is TelegateMsgType.A_MEAS_RESULT:
            # Bob receives measurement result from Alice
            for protocol in list(self.telegate_protocols):
                if (not protocol.alice and src == protocol.remote_node_name and
                        msg.bob_comm_memory_name == protocol.bob_comm_memory_name):
                    bob_comm_memory = protocol.bob_comm_memory
                    protocol.received_message(src, msg)
                    # Bob's half of the gate is done: free its resources now instead
                    # of waiting for the reservation window to time out.
                    self._early_expire(msg.reservation, bob_comm_memory)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching telegate protocol for msg from {src}")
        elif msg.msg_type is TelegateMsgType.ACK:
            # Initiator receives acknowledgment from responder - delegate Z correction to protocol
            log.logger.info(f"[telegate:{self.node.name}] Received ACK from {src}, b_bit={getattr(msg, 'b_bit', 'unknown')}")
            for protocol in list(self.telegate_protocols):
                if (protocol.alice and src == protocol.remote_node_name and
                        msg.bob_comm_memory_name == protocol.bob_comm_memory_name):
                    log.logger.info(f"[telegate:{self.node.name}] Found matching Alice protocol for ACK")
                    # Delegate Z correction to the protocol
                    alice_comm_memory = protocol.alice_comm_memory
                    b_bit = getattr(msg, "b_bit", 0)
                    protocol.alice_z_correction(b_bit)
                    # Alice's half of the gate is done: free its resources now.
                    self._early_expire(msg.reservation, alice_comm_memory)
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching telegate protocol for ACK from {src}")

    def _early_expire(self, reservation, comm_memory):
        """Release this telegate's resources the moment its gate completes.

        Expiring the RSVP rules created by ``reservation`` stops the network from
        generating further entanglement for a session that is already finished,
        and resetting the communication memory to ``RAW`` immediately returns it
        to the free pool for the next telegate.

        Because completion (not the clock) frees the resources, the reservation
        window no longer has to be sized to end right after the gate — it only
        needs to be wide enough for the gate to run.

        Args:
            reservation: Reservation whose rules should be expired.
            comm_memory: This node's communication memory used by the gate
                (Alice's or Bob's half), or ``None`` if unavailable.
        """

        now = self.node.timeline.now()
        end_t = getattr(reservation, "end_time", None)
        if end_t is not None:
            log.logger.info(f"[telegate:{self.node.name}] early expire @ t={now:,} "
                            f"(reservation end_time={end_t:,}, early_by={end_t - now:,})")

        self.node.resource_manager.expire_rules_by_reservation(reservation)
        # Free the reservation's slot in the per-memory timecards so the scheduler
        # knows the memory is available again before the window's end_time.
        self.node.network_manager.remove_reservation_from_timecards(reservation)
        # Drop this reservation's memo_to_reservation entries now, in step with the
        # rules and timecards freed above. The late end_time removal still fires but
        # is a no-op (see remove_memo_reservation_map).
        stale = [idx for idx, res in self.memo_to_reservation.items()
                 if res is reservation]
        for idx in stale:
            self.remove_memo_reservation_map(idx)
        if comm_memory is not None:
            self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)

    def _emit_complete(self, role: str, data_key: int, step: Optional[int] = None):
        """Emit completion events to all registered callbacks.

        Args:
            role: Role of the completing party ("alice" or "bob")
            data_key: Key of the data qubit
            step: Optional step number
        """
        for cb in list(self._complete_cbs):
            try:
                cb(role, data_key, step)
            except Exception as e:
                log.logger.warning(f"{self.name}: on_complete callback error: {e}")

    def gate_complete(self, role: str, data_key: int):
        """Handle gate completion.

        Called by DQCApp's wrapper and/or telegate_complete. Provides a single
        source of completion emission to avoid double-calling.

        Args:
            role: Role of the completing party ("alice" or "bob")
            data_key: Key of the data qubit
        """
        step = getattr(self, "_current_step", None)
        log.logger.info(f"[telegate:{self.node.name}] gate_complete called: role={role}, data_key={data_key}, step={step}")

        # Emit completion directly to avoid circular calls
        log.logger.info(f"[telegate:{self.node.name}] Emitting completion for role={role}, data_key={data_key}, step={step}")
        self._emit_complete(role, data_key, step)

        # Also call DQCApp completion callback if available
        if hasattr(self.node, 'dqc_app') and self.node.dqc_app:
            log.logger.info(f"[telegate:{self.node.name}] Calling DQCApp completion callback")
            if hasattr(self.node.dqc_app, '_telegate_done'):
                self.node.dqc_app._telegate_done(role, data_key, step)
            else:
                log.logger.warning(f"[telegate:{self.node.name}] DQCApp has no _telegate_done method")

    def telegate_complete(self, data_key: int, role: str = "unknown"):
        """Handle telegate operation completion.

        Called by TelegateProtocol when a telegate operation completes on either
        Alice or Bob side. Extracts the final quantum state and triggers
        completion callbacks.

        Args:
            data_key: Key of the data qubit
            role: Role of the completing party ("alice" or "bob")
        """
        log.logger.info(f"[telegate:{self.node.name}] telegate_complete called: data_key={data_key}, role={role}")
        psi = self.node.timeline.quantum_manager.get(data_key).state

        self.data_keys.append(data_key)
        self.results.append((role, data_key, psi))

        # Single emission path: through gate_complete (DQCApp wraps this to unlock)
        self.gate_complete(role, data_key)
