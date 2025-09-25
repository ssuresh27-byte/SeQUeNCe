""" TeleportApp Module
This module implements the TeleportApp class, which is responsible for managing quantum teleportation
between quantum nodes. It utilizes the TeleportProtocol to handle the teleportation process,
including the reservation of entangled pairs and the application of corrections based on classical messages.
"""

from .base_teleport_app import BaseTeleportApp
from ..entanglement_management.teleportation import TeleportProtocol, TeleportMessage
from ..topology.node import DQCNode
from ..utils import log


class TeleportApp(BaseTeleportApp):
    """Code for the teleport application.

    TeleportApp is a specialized RequestApp that implements quantum teleportation.
    It handles the teleportation protocol between two quantum nodes (Alice and Bob).

    Attributes:
        node (DQCNode): The quantum node this app is attached to.
        name (str): The name of the teleport application.
        results (list): A list of results of (timestamp, teleported_state)
        teleport_protocols (list[TeleportProtocol]): A list of teleportation protocol instances.
    """
    
    def __init__(self, node: DQCNode):
        super().__init__(node, "TeleportApp", TeleportProtocol)

    def _register_app(self, node: DQCNode, app_name: str):
        """Register the app with the node."""
        node.teleport_app = self   # register ourselves so incoming TeleportMessage lands here

    def received_message(self, src: str, msg: TeleportMessage):
        """Handle incoming teleport messages.

        Args:
            src (str): Source node name.
            msg (TeleportMessage): The teleport message received.
        """
        super().received_message(src, msg)

    def teleport_complete(self, comm_key: int):
        """Called by TeleportProtocol once Bob's qubit is corrected. comm_key holds the teleported |ψ⟩.

        Args:
            comm_key (int): The key of the comm memory where the teleported state is stored.
        """
        my_qubit = self.node.timeline.quantum_manager.get(comm_key)
        psi = my_qubit.state # get qubit state
        log.logger.info(f"{self.name}: teleport done, state={psi}")
        self.results.append((self.node.timeline.now(), psi)) # append result (timestamp, state)
