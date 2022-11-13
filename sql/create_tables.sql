CREATE TYPE status AS ENUM ('wait', 'busy', 'finish');

CREATE TABLE queue (
    id bigserial NOT NULL PRIMARY KEY,
    created_time timestamp default current_timestamp,
    user_id bigint NOT NULL,
    uuid varchar(100) NOT NULL,
    status status NOT NULL DEFAULT 'wait'
);

CREATE TABLE users (
    id bigserial NOT NULL PRIMARY KEY,
    created_time timestamp default current_timestamp,
    first_name varchar(255),
    last_name varchar(255),
    sex integer DEFAULT 0,
    bdate timestamp,
    can_access_closed bool,
    images text[],
    status status NOT NULL DEFAULT 'wait'
);

CREATE TABLE vectors (
    id bigserial NOT NULL PRIMARY KEY,
    created_time timestamp default current_timestamp,
    user_id bigint REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    vector double precision[]
);

CREATE TABLE best_vectors (
    user_id bigint REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE PRIMARY KEY,
    vector_id bigint REFERENCES vectors(id) ON DELETE CASCADE ON UPDATE CASCADE
);

----------------

-- view users with their best ids
CREATE VIEW users_with_vectors AS (
    SELECT u.*, bv.vector_id FROM users u
    LEFT JOIN best_vectors bv ON u.id = bv.user_id
);

-- drop finished rows
CREATE OR REPLACE FUNCTION clean_queue() RETURNS trigger AS $queue_cleaner$
BEGIN
    DELETE FROM queue q WHERE q.status = 'finish';
    RETURN NEW;
END;
$queue_cleaner$ LANGUAGE plpgsql;

CREATE TRIGGER queue_cleaner
AFTER UPDATE ON queue
FOR EACH ROW
EXECUTE PROCEDURE clean_queue();
