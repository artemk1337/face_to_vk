#!/bin/bash


# Main constants
PATH_TO_DIR=$(dirname "$0")
echo "Root dir: $PATH_TO_DIR"

export PYTHONPATH="$PATH_TO_DIR"

PYTHON="$PATH_TO_DIR"/../venv/bin/python
echo "Python path: $PYTHON"

PID_FILE="$PATH_TO_DIR"/pid.txt
touch $PID_FILE
echo "File with pids: $PID_FILE"

echo ""


# Run queue_processor
$PYTHON "$PATH_TO_DIR"/core/run_queue_processor.py &
QUEUE_PROCESSOR_PID=$!
echo "$QUEUE_PROCESSOR_PID" >> $PID_FILE
echo -e "queue_processor pid: $QUEUE_PROCESSOR_PID\n"


echo "Finish"
