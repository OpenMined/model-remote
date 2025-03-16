#!/bin/bash

# Configuration
SERVER_SCRIPT="llama_remote.py"
PID_FILE="/tmp/ollama_remote_server.pid"
LOG_FILE="/tmp/ollama_remote_server.log"
CONFIG_PATH=""  # Optional: Set to path of your config.json
VENV_DIR=".venv"  # Default virtual environment directory

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Determine which Python command to use
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Please install Python 3."
    exit 1
fi

# Check for foreground flag
FOREGROUND=false
if [ "$1" = "-f" ] || [ "$1" = "--foreground" ]; then
    FOREGROUND=true
    shift
fi

# If config is provided as an argument
if [ "$1" = "--config" ] && [ -n "$2" ]; then
    CONFIG_PATH="$2"
    shift 2
fi

# Determine package manager and set up virtual environment if needed
if command_exists uv; then
    # Check if we're in a virtual environment already
    if [ -z "$VIRTUAL_ENV" ]; then
        # If no virtual environment, check if our .venv directory exists
        if [ ! -d "$VENV_DIR" ]; then
            echo "Creating virtual environment with UV..."
            uv venv
        fi
        
        # Use the .venv environment
        if [ -f "$VENV_DIR/bin/activate" ]; then
            echo "Activating virtual environment..."
            source "$VENV_DIR/bin/activate"
        fi
    fi
    
    PKG_MGR="uv pip"
    RUNNER="uv run"
else
    PKG_MGR="pip"
    RUNNER=""
fi

# Check if --skip-deps flag is provided (default to installing deps)
INSTALL_DEPS=true
for arg in "$@"; do
    if [ "$arg" == "--skip-deps" ]; then
        INSTALL_DEPS=false
    fi
done

# Install dependencies by default unless --skip-deps is specified
if [ "$INSTALL_DEPS" == "true" ]; then
    echo "Installing dependencies..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "Running: $PKG_MGR install -r requirements.txt"
        $PKG_MGR install -r requirements.txt
    else
        # Install individual packages if no requirements file
        echo "Installing required packages..."
        echo "Running: $PKG_MGR install httpx pandas loguru pydantic syft_event syft_rpc"
        $PKG_MGR install httpx pandas loguru pydantic syft_event syft_rpc
    fi
fi

# If running in foreground mode, execute directly
if [ "$FOREGROUND" = true ]; then
    echo "Starting Ollama Remote server in foreground mode..."
    
    # Create clean arguments without our custom flags
    CLEAN_ARGS=()
    for arg in "$@"; do
        if [ "$arg" != "--skip-deps" ] && [ "$arg" != "--install-deps" ]; then
            CLEAN_ARGS+=("$arg")
        fi
    done
    
    # Run with config if provided
    if [ -n "$CONFIG_PATH" ]; then
        if [ -n "$RUNNER" ]; then
            echo "Running: $RUNNER $PYTHON_CMD -m llama_remote --config $CONFIG_PATH ${CLEAN_ARGS[@]}"
            $RUNNER $PYTHON_CMD -m llama_remote --config "$CONFIG_PATH" "${CLEAN_ARGS[@]}"
        else
            echo "Running: $PYTHON_CMD -m llama_remote --config $CONFIG_PATH ${CLEAN_ARGS[@]}"
            $PYTHON_CMD -m llama_remote --config "$CONFIG_PATH" "${CLEAN_ARGS[@]}"
        fi
    else
        if [ -n "$RUNNER" ]; then
            echo "Running: $RUNNER $PYTHON_CMD -m llama_remote ${CLEAN_ARGS[@]}"
            $RUNNER $PYTHON_CMD -m llama_remote "${CLEAN_ARGS[@]}"
        else
            echo "Running: $PYTHON_CMD -m llama_remote ${CLEAN_ARGS[@]}"
            $PYTHON_CMD -m llama_remote "${CLEAN_ARGS[@]}"
        fi
    fi
    exit $?
fi

# Background mode
# Check if server is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo "Server is already running with PID $PID"
        exit 0
    else
        echo "Stale PID file found. Removing..."
        rm "$PID_FILE"
    fi
fi

# Create clean arguments without our custom flags
CLEAN_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--skip-deps" ] && [ "$arg" != "--install-deps" ]; then
        CLEAN_ARGS+=("$arg")
    fi
done

# Start the server in the background
echo "Starting Ollama Remote server in background mode..."

# Start with config if provided
if [ -n "$CONFIG_PATH" ]; then
    if [ -n "$RUNNER" ]; then
        $RUNNER $PYTHON_CMD -m llama_remote --config "$CONFIG_PATH" "${CLEAN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    else
        $PYTHON_CMD -m llama_remote --config "$CONFIG_PATH" "${CLEAN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    fi
else
    if [ -n "$RUNNER" ]; then
        $RUNNER $PYTHON_CMD -m llama_remote "${CLEAN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    else
        $PYTHON_CMD -m llama_remote "${CLEAN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    fi
fi

PID=$!
echo $PID > "$PID_FILE"
echo "Server started with PID $PID. Logs at $LOG_FILE"

# Function to check if server is healthy
check_server() {
    # Wait a moment for server to start
    sleep 2
    
    # Check if process is still running
    if ! ps -p "$PID" > /dev/null; then
        echo "ERROR: Server failed to start!"
        echo "Last few log lines:"
        tail -n 10 "$LOG_FILE"
        rm "$PID_FILE"
        exit 1
    fi
    
    echo "Server appears to be running correctly."
    echo "To stop the server: kill $(cat "$PID_FILE")"
    echo "To view logs: tail -f $LOG_FILE"
}

# Check the server health (only in background mode)
check_server 