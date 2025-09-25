"""Base Teleportation App Implementation
    This module provides base classes for teleportation and teledata applications,
    eliminating code duplication between the two implementations.
"""

from abc import ABC, abstractmethod
from .request_app import RequestApp
from ..utils import log
from ..topology.node import DQCNode
from ..resource_management.memory_manager import MemoryInfo
from ..kernel.process import Process
from ..kernel.event import Event
from typing import List, Type


class BaseTeleportApp(RequestApp, ABC):
    """Base code for teleportation/teledata applications.

    BaseTeleportApp is a specialized RequestApp that implements quantum teleportation/teledata.
    It handles the teleportation protocol between two quantum nodes (Alice and Bob).

    Attributes:
        node (DQCNode): The quantum node this app is attached to.
        name (str): The name of the teleportation application.
        results (list): A list of results of (timestamp, teleported_state)
        data_keys (list): A list of data keys for the teleported states (teledata only).
        protocols (list): A list of protocol instances.
    """

    def __init__(self, node: DQCNode, app_name: str, protocol_class: Type):
        super().__init__(node)
        self.name = f"{self.node.name}.{app_name}"
        self.protocol_class = protocol_class
        self.results = []          # where we'll collect Bob's teleported state
        self.data_keys = []        # where we'll collect Bob's data keys (teledata only)
        self.protocols: List = []  # a list of protocol instances
        self._register_app(node, app_name)
        log.logger.debug(f"{self.name}: initialized")

    @abstractmethod
    def _register_app(self, node: DQCNode, app_name: str):
        """Register the app with the node. Implementation depends on app type."""
        pass

    def start(self, responder: str, start_t: int, end_t: int, memory_size: int, fidelity: float, data_memory_index: int):
        """Start the teleportation/teledata process.

        NOTE: only teleport one data memory qubit

        Args: 
            responder (str): Name of the responder node (Bob).
            start_t (int): Start time of the teleportation (in ps).
            end_t (int): End time of the teleportation (in ps).
            memory_size (int): Size of the memory used for the teleportation.
            fidelity (float): Target fidelity of the teleportation.
            data_memory_index (int): Index of the data qubit to be teleported.
        """
        log.logger.debug(f"{self.name}: start() → responder={responder}, data_memory_index={data_memory_index}")

        # reserve and generate EPR pair
        super().start(responder, start_t, end_t, memory_size, fidelity)

        # init a new protocol for Alice only, and append to the list
        protocol = self.protocol_class(self.node, alice=True, data_memory_index=data_memory_index, remote_node_name=responder)
        self.protocols.append(protocol)

    def get_reservation_result(self, reservation, result: bool):
        """Handle the reservation result from the network manager.

        Args:
            reservation (Reservation): The reservation object.
            result (bool): True if the reservation was successful, False otherwise.
        """
        super().get_reservation_result(reservation, result)
        log.logger.debug(f"{self.name}: reservation_result → {result}")

    def get_memory(self, info: MemoryInfo):
        """Handle memory state changes.

        Args:
            info (MemoryInfo): Information about the memory state change.
        """
        log.logger.debug(f"{self.name}: get_memory, name={info.memory.name}, state={info.state}")
        # once we see our entangled half, hand it to the protocol
        if info.index in self.memo_to_reservation:
            if info.state == "ENTANGLED":
                for protocol in self.protocols:
                    this_node = info.memory.owner.name
                    remote_node = info.remote_node
                    if this_node == protocol.owner.name and remote_node == protocol.remote_node_name:
                        # this node is Alice
                        protocol.set_alice_comm_memory_name(info.memory.name)
                        protocol.set_alice_comm_memory(info.memory)
                        protocol.set_bob_comm_memory_name(info.remote_memo)
                        reservation = self.memo_to_reservation[info.index]
                        # Let Bob first execute EntanglementGenerationA._entanglement_succeed(), then let Alice do the Bell measurement
                        time_now = self.node.timeline.now()
                        process = Process(protocol, 'alice_bell_measurement', [reservation])
                        priority = self.node.timeline.schedule_counter
                        event = Event(time_now, process, priority)
                        self.node.timeline.schedule(event)
                        break # if never reached this break, then go to else
                else:
                    # this node is Bob, create the new protocol instance, then append to self.protocols
                    protocol = self.protocol_class(self.node, alice=False, remote_node_name=info.remote_node)
                    protocol.set_bob_comm_memory_name(info.memory.name)
                    protocol.set_bob_comm_memory(info.memory)
                    protocol.set_alice_comm_memory_name(info.remote_memo)
                    self.protocols.append(protocol)

    def received_message(self, src: str, msg):
        """Handle incoming teleportation/teledata messages.

        Args:
            src (str): Source node name.
            msg: The teleportation/teledata message received.
        """
        log.logger.debug(f"{self.name} received_message from {src}: {msg}")
        if msg.msg_type.name == "MEASUREMENT_RESULT":  # Bob receives measurement result from Alice
            for protocol in self.protocols:   # find the correct protocol on Bob's side
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    protocol.received_message(src, msg)
                    self.node.resource_manager.expire_rules_by_reservation(msg.reservation)                    # early release of resources
                    self.node.resource_manager.update(None, protocol.bob_comm_memory, MemoryInfo.RAW) # release the bob comm memory
                    protocol.bob_acknowledge_complete(msg.reservation)
                    self.protocols.remove(protocol)  # remove the protocol instance, it's lifecycle is complete
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching protocol for msg={msg} from {src}")

        elif msg.msg_type.name == "ACK":              # Alice receives acknowledgment from Bob
            for protocol in self.protocols:  # find the correct protocol on Alice's side
                if src == protocol.remote_node_name and msg.bob_comm_memory_name == protocol.bob_comm_memory_name:
                    self.node.resource_manager.expire_rules_by_reservation(msg.reservation)                      # expire the rules
                    self.node.resource_manager.update(None, protocol.alice_comm_memory, MemoryInfo.RAW) # release the alice comm memory
                    self.protocols.remove(protocol)  # remove the protocol instance, it's lifecycle is complete
                    break
            else:
                log.logger.warning(f"{self.name}: received_message: no matching protocol for msg={msg} from {src}")

