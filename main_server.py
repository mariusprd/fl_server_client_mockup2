# main_server.py
# Copyright 2024 Google LLC. All Rights Reserved.
# ... (license text) ...
"""Main script to configure and run the Flower server using a specific transport."""

import logging
from pathlib import Path

# Flower imports
from flwr.common import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.server_config import ServerConfig

# Local imports
from transport.file_transport import FileTransport # Choose the transport implementation
from server_core import Server, TransportClientProxy # Import Server logic and Proxy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [Server] %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ---
NUM_CLIENTS = 2  # Expected number of clients
NUM_ROUNDS = 3   # Number of FL rounds
CLIENT_IDS = [f"client_{i}" for i in range(NUM_CLIENTS)]
EXCHANGE_DIR = Path("./flower_file_exchange") # Shared directory for file transport
ROUND_TIMEOUT = 120.0 # Timeout for waiting for client results in seconds

def main():
    log.info("Starting Flower Server setup...")

    # 1. Initialize Communication Transport and Client Proxies
    #    For file transport, we create one transport instance per client ID.
    #    These will be wrapped by TransportClientProxy.
    client_proxies = []
    log.info(f"Initializing FileTransport for {NUM_CLIENTS} clients in '{EXCHANGE_DIR.resolve()}'...")
    EXCHANGE_DIR.mkdir(exist_ok=True)
    # Optional: Clean previous run files
    log.info("Clearing previous files from exchange directory...")
    for item in EXCHANGE_DIR.glob('*.pkl'): item.unlink(missing_ok=True)
    for item in EXCHANGE_DIR.glob('*.flag'): item.unlink(missing_ok=True)

    transports = {} # Keep track of transports for potential cleanup
    for cid in CLIENT_IDS:
        transport = FileTransport(client_id=cid, exchange_dir=EXCHANGE_DIR)
        transports[cid] = transport
        proxy = TransportClientProxy(cid=cid, transport=transport)
        client_proxies.append(proxy)

    # 2. Initialize Client Manager and register proxies
    client_manager = SimpleClientManager()
    for proxy in client_proxies:
        client_manager.register(proxy)
        log.info(f"Registered client proxy: {proxy.cid}")

    if client_manager.num_available() != NUM_CLIENTS:
        log.warning(f"Client manager has {client_manager.num_available()} proxies, expected {NUM_CLIENTS}.")
        # Depending on the strategy, this might be okay or might cause issues.

    # 3. Initialize Strategy (using FedAvg as example)
    #    Using empty initial parameters for this example. A real scenario might
    #    load initial parameters or use strategy.initialize_parameters.
    initial_params = ndarrays_to_parameters([])
    strategy = FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS, # Wait for all registered clients
        initial_parameters=initial_params,
        # Add other strategy configurations as needed (e.g., evaluate_fn)
    )
    log.info("Initialized FedAvg strategy.")

    # 4. Initialize Server
    server = Server(client_manager=client_manager, strategy=strategy)
    server.set_max_workers(NUM_CLIENTS) # Allow concurrent communication

    # 5. Start Server Training Loop
    config = ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT)
    log.info(f"Starting server fit for {config.num_rounds} rounds...")

    try:
        history, elapsed_time = server.fit(num_rounds=config.num_rounds, timeout=config.round_timeout)
        log.info("Server fitting process finished.")
        log.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        log.info("--- History ---")
        log.info(str(history))
        log.info("---------------")

    except Exception as e:
        log.error(f"Server execution failed: {e}", exc_info=True)

    finally:
        # 6. Shutdown clients
        log.info("Instructing clients to disconnect...")
        server.disconnect_all_clients(timeout=10.0) # Short timeout for disconnect

        # 7. Cleanup transports (optional, depends on transport implementation)
        log.info("Cleaning up server-side transports...")
        for cid, transport in transports.items():
             try:
                 transport.cleanup()
             except Exception as e:
                 log.warning(f"Error cleaning up transport for {cid}: {e}")

        log.info("Flower Server finished.")

if __name__ == "__main__":
    main()
