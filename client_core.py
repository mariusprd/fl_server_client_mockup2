# client_core.py
# Copyright 2024 Google LLC. All Rights Reserved.
# ... (license text) ...
"""Core Flower Client logic and polling loop, using CommunicationTransport."""

import time
import random
import logging
from typing import Dict, Tuple, Optional, Any

import numpy as np

# Flower imports
from flwr.common import (
    Code, Status, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, ReconnectIns, DisconnectRes, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters
)

# Local transport import
from transport.base import CommunicationTransport

# Configure basic logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s Client-%(client_id)s | %(message)s")


# --- Mock Client Logic ---
# This class remains the same as before, representing the ML part.
class MockupClientLogic:
    """Placeholder for the actual machine learning client logic."""
    def __init__(self, cid: str):
        self.client_id = cid
        # Example: Initialize model parameters (using numpy for Flower compatibility)
        # Replace with your actual model loading/initialization
        self.parameters = [np.random.rand(10, 5).astype(np.float32), np.random.rand(5).astype(np.float32)]
        log.info(f"[Logic-{self.client_id}] Initialized with {len(self.parameters)} parameter arrays.")

    def get_parameters(self, config: Dict) -> Parameters:
        """Return the current local model parameters."""
        log.info(f"[Logic-{self.client_id}] get_parameters called. Config: {config}")
        return ndarrays_to_parameters(self.parameters)

    def fit(self, parameters: Parameters, config: Dict) -> Tuple[Parameters, int, Dict]:
        """Train the local model using the provided parameters."""
        log.info(f"[Logic-{self.client_id}] fit called. Config: {config}")
        received_params = parameters_to_ndarrays(parameters)
        log.info(f"[Logic-{self.client_id}] Received {len(received_params)} parameter arrays for training.")

        # --- Simulate Training ---
        # Replace this section with your actual model training loop
        log.info(f"[Logic-{self.client_id}] Simulating training...")
        # Example: Update local parameters (e.g., simple averaging simulation)
        if len(received_params) == len(self.parameters):
             for i in range(len(self.parameters)):
                 # Ensure shapes match before attempting operations
                 if self.parameters[i].shape == received_params[i].shape:
                      self.parameters[i] = (self.parameters[i] + received_params[i]) / 2.0
                 else:
                      log.warning(f"[Logic-{self.client_id}] Parameter shape mismatch at index {i}, skipping update.")
                      # In a real scenario, might need to reinitialize or handle error
        else:
             log.warning(f"[Logic-{self.client_id}] Received parameter count mismatch. Local: {len(self.parameters)}, Received: {len(received_params)}. Keeping local parameters.")

        sleep_duration = random.uniform(1, 3) # Simulate time delay
        time.sleep(sleep_duration)
        # --- End Simulation ---

        num_examples_fit = random.randint(50, 150) # Example data size
        metrics = {"fit_duration": sleep_duration, "loss_after_fit": random.uniform(0.5, 2.0)}
        log.info(f"[Logic-{self.client_id}] Fit finished. Returning {len(self.parameters)} updated parameter arrays.")
        return ndarrays_to_parameters(self.parameters), num_examples_fit, metrics

    def evaluate(self, parameters: Parameters, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the local model using the provided parameters."""
        log.info(f"[Logic-{self.client_id}] evaluate called. Config: {config}")
        received_params = parameters_to_ndarrays(parameters)
        log.info(f"[Logic-{self.client_id}] Received {len(received_params)} parameter arrays for evaluation.")

        # --- Simulate Evaluation ---
        # Replace this with your actual model evaluation logic
        log.info(f"[Logic-{self.client_id}] Simulating evaluation...")
        # Use the received parameters for evaluation
        sleep_duration = random.uniform(0.5, 2)
        time.sleep(sleep_duration)
        # Simulate metrics based on received parameters
        loss = random.uniform(0.1, 1.5) * (1 + np.mean([np.mean(p) for p in received_params if p.size > 0])) # Dummy loss calculation
        accuracy = random.uniform(0.6, 0.95) / (1 + loss) # Dummy accuracy
        # --- End Simulation ---

        num_examples_eval = random.randint(20, 50)
        metrics = {"eval_duration": sleep_duration, "accuracy": accuracy}
        log.info(f"[Logic-{self.client_id}] Evaluation finished. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), num_examples_eval, metrics


# --- Client Polling Loop ---
def run_client_loop(cid: str, transport: CommunicationTransport, client_logic: MockupClientLogic, poll_timeout: float = 60.0):
    """
    Main client loop that waits for instructions via transport and processes them.

    Args:
        cid: The client ID.
        transport: The CommunicationTransport instance to use.
        client_logic: The object containing the ML logic (get_parameters, fit, evaluate).
        poll_timeout: How long to wait for an instruction before polling again.
    """
    # Use a logger adapter to automatically include the client ID
    adapter = logging.LoggerAdapter(log, {'client_id': cid})
    adapter.info("Starting client loop...")
    running = True

    while running:
        instruction: Optional[Any] = None
        instruction_type: Optional[str] = None
        result: Optional[Any] = None

        try:
            # Wait for the next instruction from the server
            adapter.info(f"Waiting for instruction (timeout: {poll_timeout}s)...")
            instruction, instruction_type = transport.receive_instruction(timeout=poll_timeout)

            if instruction is None or instruction_type is None:
                # Timeout occurred, continue polling
                adapter.debug("Polling timeout, waiting again...")
                continue # Go to the start of the while loop

            adapter.info(f"Received instruction type: '{instruction_type}'")

            # --- Process Instruction ---
            status_code = Code.OK
            status_message = "Success"

            if isinstance(instruction, GetParametersIns):
                params = client_logic.get_parameters(instruction.config)
                result = GetParametersRes(status=Status(status_code, status_message), parameters=params)

            elif isinstance(instruction, FitIns):
                params, num_examples, metrics = client_logic.fit(instruction.parameters, instruction.config)
                result = FitRes(status=Status(status_code, status_message), parameters=params, num_examples=num_examples, metrics=metrics)

            elif isinstance(instruction, EvaluateIns):
                loss, num_examples, metrics = client_logic.evaluate(instruction.parameters, instruction.config)
                result = EvaluateRes(status=Status(status_code, status_message), loss=loss, num_examples=num_examples, metrics=metrics)

            elif isinstance(instruction, ReconnectIns):
                adapter.info("Received ReconnectIns (Shutdown signal). Preparing to exit.")
                result = DisconnectRes(reason="SHUTDOWN_REQUESTED")
                running = False # Signal loop termination

            else:
                adapter.warning(f"Received unknown instruction type: {type(instruction)}")
                status_code = Code.EXECUTION_FAILED
                status_message = f"Unknown instruction type {type(instruction)}"
                # No specific result object to send back for unknown types

            # --- Send Result (if applicable) ---
            if result and instruction_type:
                adapter.info(f"Sending result for '{instruction_type}'...")
                transport.send_result(result, instruction_type)
                adapter.info("Result sent.")

        except Exception as e:
            adapter.error(f"Error during client loop processing instruction '{instruction_type}': {e}", exc_info=True)
            # Attempt to send an error status back if possible and appropriate
            # This depends heavily on the specific error and transport capabilities
            if instruction_type and instruction_type in ["fit", "evaluate", "get_parameters"]:
                 error_result = None
                 status = Status(Code.EXECUTION_FAILED, f"Client-side error: {e}")
                 if instruction_type == "fit": error_result = FitRes(status=status, parameters=Parameters([], ""), num_examples=0, metrics={})
                 elif instruction_type == "evaluate": error_result = EvaluateRes(status=status, loss=0.0, num_examples=0, metrics={})
                 elif instruction_type == "get_parameters": error_result = GetParametersRes(status=status, parameters=Parameters([], ""))

                 if error_result:
                     try:
                         adapter.warning(f"Attempting to send error result for '{instruction_type}'...")
                         transport.send_result(error_result, instruction_type)
                     except Exception as send_err:
                         adapter.error(f"Failed to send error result: {send_err}")
            # Consider if the client should exit on certain errors
            # running = False # Uncomment to exit on any error

        # Optional short pause to prevent tight looping on errors/timeouts
        if instruction is None:
             time.sleep(1)

    adapter.info("Client loop finished.")
    # Perform cleanup using the transport
    transport.cleanup()
