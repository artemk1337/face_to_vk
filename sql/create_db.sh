#!/bin/bash

USERNAME="admin"
PASSWORD="admin"
HOST="127.0.0.1"
PORT="5432"
DB="postgres"

CREATE_FILENAME="./create_tables.sql"


psql postgres://$USERNAME:$PASSWORD@$HOST:$PORT/$DB << EOF
  CREATE DATABASE face2vk;
EOF

bash run_sql.sh "$CREATE_FILENAME"
