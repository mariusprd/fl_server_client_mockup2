# main_client.py
# Copyright 2024 Google LLC. All Rights Reserved.
# ... (license text) ...
"""Main script to run a Flower client instance using a specific transport."""

import argparse
import logging
from pathlib import Path

# Local imports
from transport.file_transport import FileTransport # Choose the transport
from client_core import MockupClientLogic, run_client_loop # Import client logic and loop

# Configure logging (the loop uses an adapter to add client ID)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ---
EXCHANGE_DIR = Path("./flower_file_exchange") # Shared directory for file transport
POLL_TIMEOUT = 60.0 # How long client waits for one instruction

def main():
    parser = argparse.ArgumentParser(description="Flower File-Based Client Runner")
    parser.add_argument(
        "--cid", type=str, required=True, help="Client ID (e.g., client_0)"
    )
    args = parser.parse_args()
    cid = args.cid

    log.info(f"Starting client {cid}...")

    # 1. Initialize Communication Transport
    #    Use the same transport type and configuration as the server.
    log.info(f"Initializing FileTransport in '{EXCHANGE_DIR.resolve()}'")
    EXCHANGE_DIR.mkdir(exist_ok=True) # Ensure directory exists
    transport = FileTransport(client_id=cid, exchange_dir=EXCHANGE_DIR)

    # 2. Initialize Client Logic
    #    Replace MockupClientLogic with your actual client implementation.
    client_logic = MockupClientLogic(cid=cid)
    log.info("Initialized MockupClientLogic.")

    # 3. Start Client Loop
    log.info("Starting client processing loop...")
    try:
        run_client_loop(
            cid=cid,
            transport=transport,
            client_logic=client_logic,
            poll_timeout=POLL_TIMEOUT
        )
    except Exception as e:
        log.error(f"Client {cid} execution failed: {e}", exc_info=True)
    finally:
        log.info(f"Client {cid} finished.")
        # Cleanup is handled inside run_client_loop calling transport.cleanup()

if __name__ == "__main__":
    main()
