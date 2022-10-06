#!/bin/bash

USERNAME="admin"
PASSWORD="admin"
HOST="127.0.0.1"
PORT="5432"
DB="postgres"

DROP_FILENAME="./drop_tables.sql"


bash run_sql.sh "$DROP_FILENAME"

psql postgres://$USERNAME:$PASSWORD@$HOST:$PORT/$DB << EOF
  DROP DATABASE face2vk;
EOF
