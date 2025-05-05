# transport/base.py
# Copyright 2024 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Abstract Base Class for Server-Client Communication Transport."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

class CommunicationTransport(ABC):
    """
    Abstract interface for sending/receiving messages between server and client.

    This allows swapping the underlying communication mechanism (e.g., files,
    gRPC, REST, ZeroMQ) without changing the core server/client logic significantly.
    """

    @abstractmethod
    def send_instruction(self, instruction: Any, instruction_type: str) -> None:
        """
        Sends an instruction object from the server to a specific client.

        Args:
            instruction: The instruction object (e.g., FitIns, EvaluateIns).
            instruction_type: A string identifying the type (e.g., "fit", "evaluate").
        """
        pass

    @abstractmethod
    def receive_instruction(self, timeout: Optional[float]) -> Tuple[Optional[Any], Optional[str]]:
        """
        Receives the next instruction object sent to this client by the server.

        This is a blocking call on the client side until an instruction arrives
        or a timeout occurs.

        Args:
            timeout: Maximum time in seconds to wait for an instruction.

        Returns:
            A tuple containing:
            - The received instruction object (or None if timeout/error).
            - The instruction type string (or None if timeout/error).
        """
        pass

    @abstractmethod
    def send_result(self, result: Any, instruction_type: str) -> None:
        """
        Sends a result object from the client back to the server.

        Args:
            result: The result object (e.g., FitRes, EvaluateRes).
            instruction_type: The type string of the instruction this result
                              corresponds to (e.g., "fit", "evaluate"). This helps
                              the server match results to requests.
        """
        pass

    @abstractmethod
    def receive_result(self, instruction_type: str, timeout: Optional[float]) -> Any:
        """
        Receives the result object sent by a specific client for a given instruction type.

        This is a blocking call on the server side.

        Args:
            instruction_type: The type string of the instruction for which a
                              result is expected.
            timeout: Maximum time in seconds to wait for the result.

        Returns:
            The received result object (or raises TimeoutError/Exception on failure).
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Perform any necessary cleanup for this transport (e.g., delete files,
        close connections). Called on shutdown.
        """
        pass
