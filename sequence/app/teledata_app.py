""" TeledataApp Module
This module implements the TeledataApp class, which is responsible for managing quantum teledata
between quantum nodes. It utilizes the TeledataProtocol to handle the teledata process,
including the reservation of entangled pairs and the application of corrections based on classical messages.
"""

from .base_teleport_app import BaseTeleportApp
from ..entanglement_management.teleportation import TeledataProtocol, TeledataMessage
from ..topology.node import DQCNode
from ..utils import log


class TeledataApp(BaseTeleportApp):
    """Code for the teledata application.

    TeledataApp is a specialized RequestApp that implements quantum teledata.
    It handles the teledata protocol between two quantum nodes (Alice and Bob).

    Attributes:
        node (DQCNode): The quantum node this app is attached to.
        name (str): The name of the teledata application.
        results (list): A list of results of (timestamp, teleported_state)
        data_keys (list): A list of data keys for the teleported states.
        teledata_protocols (list[TeledataProtocol]): A list of teledata protocol instances.
    """

    def __init__(self, node: DQCNode):
        super().__init__(node, "TeledataApp", TeledataProtocol)

    def _register_app(self, node: DQCNode, app_name: str):
        """Register the app with the node."""
        node.teledata_app = self   # register ourselves so incoming TeledataMessage lands here

    def received_message(self, src: str, msg: TeledataMessage):
        """Handle incoming teledata messages.

        Args:
            src (str): Source node name.
            msg (TeledataMessage): The teledata message received.
        """
        super().received_message(src, msg)

    def teledata_complete(self, comm_key: int):
        """Called by TeledataProtocol once Bob's qubit is corrected. comm_key holds the teleported |ψ⟩.

        Args:
            comm_key (int): The key of the comm memory where the teleported state is stored.
        """
        my_qubit = self.node.timeline.quantum_manager.get(comm_key)
        psi = my_qubit.state # get qubit state
        
        # Extract the first 2 qubits if we have a 4-qubit state (due to SWAP operation)
        if len(psi) == 4:
            psi = psi[:2]
        
        log.logger.info(f"{self.name}: teledata done, state={psi}")
        self.data_keys.append(comm_key)
        self.results.append((self.node.timeline.now(), psi)) # append result (timestamp, state)
