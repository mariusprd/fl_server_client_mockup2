# transport/file_transport.py
# Copyright 2024 Google LLC. All Rights Reserved.
# ... (license text) ...
"""File-based implementation of the CommunicationTransport interface."""

import pickle
import time
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from .base import CommunicationTransport

# Configure basic logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s")


class FileTransport(CommunicationTransport):
    """
    Implements CommunicationTransport using temporary files and polling.

    WARNING: Inefficient, not robust, insecure with untrusted data (pickle).
             Suitable only for basic demonstration or local testing.
    """
    def __init__(self, client_id: str, exchange_dir: Path, poll_interval: float = 0.2):
        self.client_id = client_id
        self.exchange_dir = Path(exchange_dir)
        self.poll_interval = poll_interval
        self.exchange_dir.mkdir(exist_ok=True) # Ensure directory exists
        log.debug(f"[FileTransport-{self.client_id}] Initialized. Dir: {self.exchange_dir.resolve()}")

    def _get_paths(self, instruction_type: str, direction: str) -> Tuple[Path, Path]:
        """Helper to get file and flag paths."""
        if direction == "instruction":
            file_path = self.exchange_dir / f"{self.client_id}_{instruction_type}.instruction.pkl"
            flag_path = self.exchange_dir / f"{self.client_id}_{instruction_type}.instruction.flag"
        elif direction == "result":
            file_path = self.exchange_dir / f"{self.client_id}_{instruction_type}.result.pkl"
            flag_path = self.exchange_dir / f"{self.client_id}_{instruction_type}.result.flag"
        else:
            raise ValueError("Direction must be 'instruction' or 'result'")
        return file_path, flag_path

    def send_instruction(self, instruction: Any, instruction_type: str) -> None:
        """Sends instruction via file."""
        file_path, flag_path = self._get_paths(instruction_type, "instruction")
        log.debug(f"[FileTransport-{self.client_id}] Sending instruction '{instruction_type}' to {file_path}")
        try:
            # Clean up previous potentially stale files for this exact instruction
            file_path.unlink(missing_ok=True)
            flag_path.unlink(missing_ok=True)
            time.sleep(0.01) # Small delay before writing

            # Write data first, then flag
            with open(file_path, "wb") as f:
                pickle.dump(instruction, f)
            flag_path.touch() # Create empty flag file
            log.debug(f"[FileTransport-{self.client_id}] Instruction '{instruction_type}' written.")
        except Exception as e:
            log.error(f"[FileTransport-{self.client_id}] Error writing instruction file '{file_path}': {e}")
            # Clean up potentially corrupted files
            file_path.unlink(missing_ok=True)
            flag_path.unlink(missing_ok=True)
            raise # Re-raise to signal failure

    def receive_instruction(self, timeout: Optional[float]) -> Tuple[Optional[Any], Optional[str]]:
        """Polls for and receives the next instruction file."""
        log.debug(f"[FileTransport-{self.client_id}] Waiting for next instruction...")
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else 3600 # Default to long wait if None

        while time.time() - start_time < effective_timeout:
            # Check for any instruction flag file for this client
            flag_files = list(self.exchange_dir.glob(f"{self.client_id}_*.instruction.flag"))

            if flag_files:
                instruction_flag_path = flag_files[0] # Process one at a time
                base_name = instruction_flag_path.name.replace(".instruction.flag", "")
                instruction_file_path = self.exchange_dir / f"{base_name}.instruction.pkl"
                instruction_type = base_name.replace(f"{self.client_id}_", "")

                log.debug(f"[FileTransport-{self.client_id}] Found instruction flag: {instruction_flag_path.name}")

                try:
                    # Wait a tiny moment to ensure file write is complete before reading
                    time.sleep(0.05)
                    with open(instruction_file_path, "rb") as f:
                        instruction = pickle.load(f)
                    log.debug(f"[FileTransport-{self.client_id}] Read instruction '{instruction_type}'.")

                    # Delete instruction files AFTER successful read
                    instruction_file_path.unlink(missing_ok=True)
                    instruction_flag_path.unlink(missing_ok=True)
                    log.debug(f"[FileTransport-{self.client_id}] Deleted instruction files for '{instruction_type}'.")
                    return instruction, instruction_type # Success

                except FileNotFoundError:
                    log.warning(f"[FileTransport-{self.client_id}] Flag {instruction_flag_path.name} existed, but file {instruction_file_path.name} not found. Cleaning flag.")
                    instruction_flag_path.unlink(missing_ok=True)
                    # Continue polling
                except Exception as e:
                    log.error(f"[FileTransport-{self.client_id}] Error reading/deleting instruction file {instruction_file_path.name if instruction_file_path else 'N/A'}: {e}")
                    # Attempt cleanup
                    if instruction_file_path and instruction_file_path.exists(): instruction_file_path.unlink(missing_ok=True)
                    if instruction_flag_path and instruction_flag_path.exists(): instruction_flag_path.unlink(missing_ok=True)
                    return None, None # Indicate error

            # No instruction found yet, wait before polling again
            time.sleep(self.poll_interval)

        log.debug(f"[FileTransport-{self.client_id}] Timeout waiting for instruction.")
        return None, None # Timeout

    def send_result(self, result: Any, instruction_type: str) -> None:
        """Sends result via file."""
        file_path, flag_path = self._get_paths(instruction_type, "result")
        log.debug(f"[FileTransport-{self.client_id}] Sending result for '{instruction_type}' to {file_path}")
        try:
             # Clean up previous potentially stale files
            file_path.unlink(missing_ok=True)
            flag_path.unlink(missing_ok=True)
            time.sleep(0.01)

            # Write data first, then flag
            with open(file_path, "wb") as f:
                pickle.dump(result, f)
            flag_path.touch()
            log.debug(f"[FileTransport-{self.client_id}] Result for '{instruction_type}' written.")
        except Exception as e:
            log.error(f"[FileTransport-{self.client_id}] Error writing result file '{file_path}': {e}")
            # Clean up potentially corrupted files
            file_path.unlink(missing_ok=True)
            flag_path.unlink(missing_ok=True)
            raise # Re-raise to signal failure

    def receive_result(self, instruction_type: str, timeout: Optional[float]) -> Any:
        """Polls for and receives a specific result file."""
        result_file_path, result_flag_path = self._get_paths(instruction_type, "result")
        log.debug(f"[FileTransport-{self.client_id}] Waiting for result '{instruction_type}' from {result_flag_path}...")
        effective_timeout = timeout if timeout is not None else 120 # Default timeout

        start_time = time.time()
        while time.time() - start_time < effective_timeout:
            if result_flag_path.exists():
                log.debug(f"[FileTransport-{self.client_id}] Result flag found: {result_flag_path}")
                try:
                    # Wait a tiny moment to ensure file write is complete
                    time.sleep(0.05)
                    with open(result_file_path, "rb") as f:
                        result = pickle.load(f)

                    # Cleanup result files AFTER successful read
                    result_file_path.unlink(missing_ok=True)
                    result_flag_path.unlink(missing_ok=True)
                    log.debug(f"[FileTransport-{self.client_id}] Read and deleted result files for '{instruction_type}'.")
                    return result # Success
                except FileNotFoundError:
                     log.warning(f"[FileTransport-{self.client_id}] Result flag existed but .pkl file disappeared: {result_file_path}. Retrying poll.")
                     # Flag existed, but file gone? Maybe client cleanup race condition? Continue polling briefly.
                     time.sleep(self.poll_interval)
                except Exception as e:
                    log.error(f"[FileTransport-{self.client_id}] Error reading/deleting result file '{result_file_path}': {e}")
                    # Cleanup potentially corrupted files
                    result_file_path.unlink(missing_ok=True)
                    result_flag_path.unlink(missing_ok=True)
                    raise # Re-raise the exception to signal failure
            else:
                time.sleep(self.poll_interval)

        log.warning(f"[FileTransport-{self.client_id}] Timeout waiting for result flag: {result_flag_path}")
        raise TimeoutError(f"Client {self.client_id} did not provide result for '{instruction_type}' within {effective_timeout}s")

    def cleanup(self) -> None:
        """Removes files associated with this client."""
        log.info(f"[FileTransport-{self.client_id}] Cleaning up exchange files...")
        count = 0
        for f in self.exchange_dir.glob(f"{self.client_id}_*"):
            try:
                f.unlink()
                count += 1
            except OSError as e:
                log.warning(f"[FileTransport-{self.client_id}] Error deleting file {f}: {e}")
        log.info(f"[FileTransport-{self.client_id}] Deleted {count} files.")

