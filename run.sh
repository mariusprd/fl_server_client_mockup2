#!/bin/bash

# --- Configuration ---
NUM_CLIENTS=2 # Must match main_server.py NUM_CLIENTS
EXCHANGE_DIR="./flower_file_exchange"
SERVER_LOG="server.log"
CLIENT_LOG_PREFIX="client"

# --- Setup ---
echo "Refactored Flower Run Script"
echo "----------------------------"

# Create exchange directory
mkdir -p $EXCHANGE_DIR
echo "Ensured exchange directory exists: $EXCHANGE_DIR"

# Optional: Clean previous run logs and exchange files
echo "Clearing previous logs and exchange files..."
rm -f $SERVER_LOG
rm -f ${CLIENT_LOG_PREFIX}_*.log
rm -f $EXCHANGE_DIR/*.pkl
rm -f $EXCHANGE_DIR/*.flag

# --- Start Server ---
echo "Starting server in background (logging to console and $SERVER_LOG)..."
# Use tee to send output to both console (stdout/stderr) and the log file
# 2>&1 redirects stderr to stdout, which is then piped to tee
python main_server.py 2>&1 | tee $SERVER_LOG &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Give the server a moment to initialize (adjust if needed)
sleep 4

# --- Start Clients ---
CLIENT_PIDS=()
echo "Starting $NUM_CLIENTS clients in background..."
for i in $(seq 0 $(($NUM_CLIENTS - 1)))
do
  CLIENT_ID="client_$i"
  CLIENT_LOG="${CLIENT_LOG_PREFIX}_${CLIENT_ID}.log"
  echo "Starting client $CLIENT_ID (logging to console and $CLIENT_LOG)..."
  # Use tee for client logs as well
  python main_client.py --cid $CLIENT_ID 2>&1 | tee $CLIENT_LOG &
  CLIENT_PID=$!
  CLIENT_PIDS+=($CLIENT_PID)
  echo "Client $CLIENT_ID PID: $CLIENT_PID"
  sleep 0.5 # Stagger client starts slightly
done

# --- Wait for Server ---
echo "----------------------------"
echo "Server and clients started."
echo "Waiting for server (PID: $SERVER_PID) to complete..."
wait $SERVER_PID
SERVER_EXIT_CODE=$?
echo "Server process finished with exit code $SERVER_EXIT_CODE."

# --- Wait for Clients (Optional) ---
echo "Waiting a few seconds for clients to finish..."
sleep 5

# --- Check Client Logs ---
echo "Check log files ($SERVER_LOG, ${CLIENT_LOG_PREFIX}_*.log) for full details."

# --- Optional: Terminate lingering clients ---
# echo "Checking for lingering client processes..."
# for pid in "${CLIENT_PIDS[@]}"; do
#   if ps -p $pid > /dev/null; then
#     echo "Client PID $pid is still running. Terminating."
#     kill $pid
#   fi
# done

echo "----------------------------"
echo "Run finished."
