# server_core.py
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
"""Core Flower Server logic, decoupled from specific transport via TransportClientProxy."""

import concurrent.futures
import io
import timeit
import logging # Import Python's standard logging
from typing import Optional, Union, List, Tuple, Dict, cast

# Flower imports
from flwr.common import (
    Code, Status, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, ReconnectIns, DisconnectRes, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy # Base class for proxy
from flwr.server.strategy import Strategy, FedAvg
from flwr.server.history import History
from flwr.server.server_config import ServerConfig

# Local transport import
from transport.base import CommunicationTransport

# Configure basic logging (will print to stderr by default)
# Ensure this is configured before any log messages are emitted
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s")
# Get the logger instance for this module
core_logger = logging.getLogger(__name__)


# --- TransportClientProxy ---
class TransportClientProxy(ClientProxy):
    """
    A ClientProxy implementation that uses a CommunicationTransport object
    to interact with the actual client.
    """
    def __init__(self, cid: str, transport: CommunicationTransport):
        super().__init__(cid)
        self.transport = transport
        # Use the correct logger object
        core_logger.debug(f"[TransportProxy {self.cid}] Initialized.")

    def get_properties(self, ins, timeout: Optional[float], group_id: Optional[int]) -> None:
        # Properties not typically handled by basic transports
        core_logger.warning(f"[TransportProxy {self.cid}] get_properties not implemented for this transport.")
        pass

    def get_parameters(self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]) -> GetParametersRes:
        """Gets parameters from the client using the transport."""
        core_logger.debug(f"[TransportProxy {self.cid}] Sending GetParametersIns.")
        try:
            self.transport.send_instruction(ins, "get_parameters")
            result = self.transport.receive_result("get_parameters", timeout)
            if isinstance(result, GetParametersRes):
                core_logger.debug(f"[TransportProxy {self.cid}] Received GetParametersRes.")
                return result
            else:
                core_logger.error(f"[TransportProxy {self.cid}] Received unexpected type for GetParametersRes: {type(result)}")
                return GetParametersRes(status=Status(Code.EXECUTION_FAILED, "Invalid result type"), parameters=Parameters([], ""))
        except Exception as e:
            core_logger.error(f"[TransportProxy {self.cid}] Failed get_parameters: {e}", exc_info=True) # Add traceback
            return GetParametersRes(status=Status(Code.EXECUTION_FAILED, str(e)), parameters=Parameters([], ""))

    def fit(self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]) -> FitRes:
        """Sends FitIns and receives FitRes via the transport."""
        core_logger.debug(f"[TransportProxy {self.cid}] Sending FitIns.")
        try:
            self.transport.send_instruction(ins, "fit")
            result = self.transport.receive_result("fit", timeout)
            if isinstance(result, FitRes):
                core_logger.debug(f"[TransportProxy {self.cid}] Received FitRes.")
                return result
            else:
                core_logger.error(f"[TransportProxy {self.cid}] Received unexpected type for FitRes: {type(result)}")
                return FitRes(status=Status(Code.EXECUTION_FAILED, "Invalid result type"), parameters=Parameters([], ""), num_examples=0, metrics={})
        except Exception as e:
            core_logger.error(f"[TransportProxy {self.cid}] Failed fit: {e}", exc_info=True) # Add traceback
            return FitRes(status=Status(Code.EXECUTION_FAILED, str(e)), parameters=Parameters([], ""), num_examples=0, metrics={})

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]) -> EvaluateRes:
        """Sends EvaluateIns and receives EvaluateRes via the transport."""
        core_logger.debug(f"[TransportProxy {self.cid}] Sending EvaluateIns.")
        try:
            self.transport.send_instruction(ins, "evaluate")
            result = self.transport.receive_result("evaluate", timeout)
            if isinstance(result, EvaluateRes):
                core_logger.debug(f"[TransportProxy {self.cid}] Received EvaluateRes.")
                return result
            else:
                core_logger.error(f"[TransportProxy {self.cid}] Received unexpected type for EvaluateRes: {type(result)}")
                return EvaluateRes(status=Status(Code.EXECUTION_FAILED, "Invalid result type"), loss=0.0, num_examples=0, metrics={})
        except Exception as e:
            core_logger.error(f"[TransportProxy {self.cid}] Failed evaluate: {e}", exc_info=True) # Add traceback
            return EvaluateRes(status=Status(Code.EXECUTION_FAILED, str(e)), loss=0.0, num_examples=0, metrics={})

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]) -> DisconnectRes:
        """Sends ReconnectIns (typically for shutdown) via the transport."""
        core_logger.debug(f"[TransportProxy {self.cid}] Sending ReconnectIns (shutdown).")
        try:
            self.transport.send_instruction(ins, "reconnect")
            # Don't always expect a result for disconnect, client might just exit.
            result = self.transport.receive_result("reconnect", timeout=5.0) # Short timeout
            if isinstance(result, DisconnectRes):
                 core_logger.debug(f"[TransportProxy {self.cid}] Received DisconnectRes.")
                 return result
            else:
                 core_logger.warning(f"[TransportProxy {self.cid}] Received unexpected type for DisconnectRes: {type(result)}")
                 return DisconnectRes(reason="INVALID_TYPE_RECEIVED")
        except TimeoutError:
            core_logger.info(f"[TransportProxy {self.cid}] No DisconnectRes received (client likely exited). Assuming success.")
            return DisconnectRes(reason="CLIENT_EXIT_NO_RES")
        except Exception as e:
            core_logger.error(f"[TransportProxy {self.cid}] Failed reconnect: {e}", exc_info=True) # Add traceback
            return DisconnectRes(reason=f"RECONNECT_TRANSPORT_ERROR: {e}")

# --- Helper Functions (Adapted from Flower source, now use ClientProxy base class) ---

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout, group_id)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(future=future, results=results, failures=failures)
    return results, failures

def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    core_logger.debug(f"Executing fit_client for {client.cid}") # Use logger
    res = client.fit(ins, timeout=timeout, group_id=group_id)
    core_logger.debug(f"Finished fit_client for {client.cid}") # Use logger
    return client, res

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    failure = future.exception()
    if failure is not None:
        core_logger.warning(f"Fit future failed: {failure}", exc_info=True) # Use logger
        failures.append(failure)
        return
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result
    if res.status.code == Code.OK:
        results.append(result)
    else:
        core_logger.warning(f"Fit future completed but resulted in status {res.status.code}: {res.status.message}") # Use logger
        failures.append(result)

def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout, group_id)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(future=future, results=results, failures=failures)
    return results, failures

def evaluate_client(
    client: ClientProxy, ins: EvaluateIns, timeout: Optional[float], group_id: int
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    core_logger.debug(f"Executing evaluate_client for {client.cid}") # Use logger
    res = client.evaluate(ins, timeout=timeout, group_id=group_id)
    core_logger.debug(f"Finished evaluate_client for {client.cid}") # Use logger
    return client, res

def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    failure = future.exception()
    if failure is not None:
        core_logger.warning(f"Evaluate future failed: {failure}", exc_info=True) # Use logger
        failures.append(failure)
        return
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result
    if res.status.code == Code.OK:
        results.append(result)
    else:
        core_logger.warning(f"Evaluate future completed but resulted in status {res.status.code}: {res.status.message}") # Use logger
        failures.append(result)

def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            core_logger.warning(f"Reconnect future failed: {failure}", exc_info=True) # Use logger
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def reconnect_client(
    client: ClientProxy, reconnect: ReconnectIns, timeout: Optional[float]
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    core_logger.debug(f"Executing reconnect_client for {client.cid}") # Use logger
    res = client.reconnect(reconnect, timeout=timeout, group_id=None)
    core_logger.debug(f"Finished reconnect_client for {client.cid}") # Use logger
    return client, res


# --- Core Server Class (Adapted from Flower source) ---
class Server:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager, # Expects ClientProxy instances
        strategy: Optional[Strategy] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(tensors=[], tensor_type="")
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        core_logger.info("Server initialized.") # Use the correct logger
        if not isinstance(self.strategy, Strategy):
            core_logger.warning("Provided strategy is not an instance of flwr.server.strategy.Strategy")
        if not isinstance(self._client_manager, ClientManager):
             core_logger.warning("Provided client_manager is not an instance of flwr.server.client_manager.ClientManager")


    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers
        core_logger.info(f"Max workers set to: {max_workers}") # Use logger

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy
        core_logger.info(f"Strategy set to: {type(strategy).__name__}") # Use logger

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run federated learning for a number of rounds."""
        history = History()

        # Initialize parameters
        core_logger.info("[INIT] Initializing parameters...")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        core_logger.info(f"Initial parameters: type={self.parameters.tensor_type}, num_tensors={len(self.parameters.tensors)}")

        # Evaluate initial parameters (if strategy supports it)
        core_logger.info("[INIT] Evaluating initial parameters (centralized)...")
        res_init_eval = self.strategy.evaluate(0, parameters=self.parameters)
        if res_init_eval is not None:
            loss, metrics = res_init_eval
            core_logger.info(f"Initial parameters (loss, metrics): {loss}, {metrics}")
            history.add_loss_centralized(server_round=0, loss=loss)
            history.add_metrics_centralized(server_round=0, metrics=metrics)
        else:
            core_logger.info("Initial centralized evaluation failed or strategy does not implement it.")

        # Run federated learning rounds
        core_logger.info(f"Starting FL rounds: {num_rounds}")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Fit round
            core_logger.info(f"\n======== [ROUND {current_round}/{num_rounds}] ========")
            core_logger.info("Starting fit round...")
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                params_prime, metrics_fit, (_, failures) = res_fit # Adjusted tuple unpacking
                if params_prime:
                    self.parameters = params_prime
                    core_logger.info("Parameter aggregation successful.")
                else:
                    core_logger.warning("Parameter aggregation failed.")
                core_logger.info(f"Fit metrics aggregated: {metrics_fit}")
                core_logger.info(f"Fit failures: {len(failures)}")
                history.add_metrics_distributed_fit(server_round=current_round, metrics=metrics_fit)
            else:
                core_logger.warning("Fit round cancelled or failed.")

            # Centralized evaluation
            core_logger.info("Starting centralized evaluation...")
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen:
                loss_cen, metrics_cen = res_cen
                core_logger.info(f"Centralized evaluation: loss={loss_cen}, metrics={metrics_cen}")
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)
            else:
                core_logger.info("Centralized evaluation failed or strategy does not implement it.")

            # Distributed evaluation
            core_logger.info("Starting distributed evaluation...")
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, metrics_fed, (_, failures) = res_fed # Adjusted tuple unpacking
                if loss_fed is not None:
                    core_logger.info(f"Distributed evaluation: loss={loss_fed}, metrics={metrics_fed}")
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=metrics_fed)
                else:
                    core_logger.warning("Distributed evaluation aggregation failed to return loss.")
                core_logger.info(f"Evaluate failures: {len(failures)}")
            else:
                core_logger.warning("Distributed evaluation cancelled or failed.")

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        core_logger.info(f"\n======== [SUMMARY] ========")
        core_logger.info(f"FL finished in {elapsed:.2f}s")
        # Use io.StringIO to format history for logging
        history_str = io.StringIO()
        print(history, file=history_str) # Print history to the string buffer
        history_str.seek(0) # Rewind buffer
        core_logger.info(f"History:\n{history_str.read()}") # Log formatted history
        return history, elapsed

    def evaluate_round(
        self, server_round: int, timeout: Optional[float]
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        core_logger.info(f"[Round {server_round}] Configuring evaluation...") # Use logger
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            core_logger.info("configure_evaluate: No clients selected, skipping evaluation.") # Use logger
            return None
        core_logger.info(f"configure_evaluate: Strategy sampled {len(client_instructions)} clients.") # Use logger

        # Collect results
        core_logger.info(f"[Round {server_round}] Collecting evaluation results...") # Use logger
        results, failures = evaluate_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        core_logger.info(f"aggregate_evaluate: Received {len(results)} results and {len(failures)} failures.") # Use logger

        # Aggregate results
        core_logger.info(f"[Round {server_round}] Aggregating evaluation results...") # Use logger
        aggregated_result = self.strategy.aggregate_evaluate(
            server_round, results, failures
        )
        if aggregated_result is None:
            core_logger.warning("aggregate_evaluate returned None") # Use logger
            return None
        loss_aggregated, metrics_aggregated = aggregated_result
        core_logger.info(f"[Round {server_round}] Aggregated evaluate results: Loss={loss_aggregated}") # Use logger
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self, server_round: int, timeout: Optional[float]
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Perform a single round of federated averaging."""
        core_logger.info(f"[Round {server_round}] Configuring fit...") # Use logger
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            core_logger.info("configure_fit: No clients selected, skipping fit round.") # Use logger
            return None
        core_logger.info(f"configure_fit: Strategy sampled {len(client_instructions)} clients.") # Use logger

        # Collect results
        core_logger.info(f"[Round {server_round}] Collecting fit results...") # Use logger
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        core_logger.info(f"aggregate_fit: Received {len(results)} results and {len(failures)} failures.") # Use logger

        # Aggregate results
        core_logger.info(f"[Round {server_round}] Aggregating fit results...") # Use logger
        aggregated_result = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        if aggregated_result is None:
             core_logger.warning("aggregate_fit returned None") # Use logger
             return None # Strategy decided not to update parameters
        parameters_aggregated, metrics_aggregated = aggregated_result
        core_logger.info(f"[Round {server_round}] Aggregated fit results obtained.") # Use logger
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        core_logger.info("Requesting disconnect from all clients...") # Use logger
        all_clients = self._client_manager.all()
        if not all_clients:
            core_logger.info("No clients connected to disconnect.") # Use logger
            return
        client_proxies = list(all_clients.values())
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(cp, instruction) for cp in client_proxies]

        core_logger.info(f"Sending disconnect to {len(client_instructions)} clients...") # Use logger
        results, failures = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        core_logger.info(f"Disconnect results: {len(results)}, failures: {len(failures)}") # Use logger

    def _get_initial_parameters(self, server_round: int, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from strategy or a client."""
        core_logger.info("Attempting to get initial parameters from strategy...") # Use logger
        parameters = self.strategy.initialize_parameters(client_manager=self._client_manager)
        if parameters is not None:
            core_logger.info("Using initial parameters provided by strategy.") # Use logger
            return parameters

        core_logger.info("Requesting initial parameters from one random client.") # Use logger
        random_clients = self._client_manager.sample(1)
        if not random_clients:
            core_logger.error("Cannot get initial parameters: No clients available!") # Use logger
            return Parameters(tensors=[], tensor_type="")

        random_client = cast(ClientProxy, random_clients[0])
        core_logger.info(f"Requesting parameters from client {random_client.cid}") # Use logger
        get_parameters_res = random_client.get_parameters(
            ins=GetParametersIns(config={}), timeout=timeout, group_id=server_round
        )
        if get_parameters_res.status.code == Code.OK:
            core_logger.info(f"Received initial parameters from client {random_client.cid}.") # Use logger
            return get_parameters_res.parameters
        else:
            core_logger.error(f"Failed to receive initial parameters from client {random_client.cid}: {get_parameters_res.status}") # Use logger
            return Parameters(tensors=[], tensor_type="")
