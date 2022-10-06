#!/bin/bash

USERNAME="admin"
PASSWORD="admin"
HOST="127.0.0.1"
PORT="5432"
DB="face2vk"

for arg in "$@"
do
  echo "Running $arg ..."
  psql postgres://$USERNAME:$PASSWORD@$HOST:$PORT/$DB -f "$arg"
done
