#!/bin/bash


# Main constants
PATH_TO_DIR=$(dirname "$0")
echo "Root dir: $PATH_TO_DIR"

PID_FILE="$PATH_TO_DIR"/pid.txt
touch $PID_FILE
echo "File with pids: $PID_FILE"

echo ""


# Kill processes
while read -r line
do
  echo -e "Kill pid $line\n"
  kill "$line"
done < "$PID_FILE"

rm $PID_FILE
echo "Finish"
