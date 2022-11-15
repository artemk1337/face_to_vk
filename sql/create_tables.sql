CREATE TYPE status AS ENUM ('wait', 'busy', 'finish');

CREATE TABLE queue (
    id serial NOT NULL PRIMARY KEY,
    created_time timestamp default current_timestamp,
    user_id bigserial NOT NULL,
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
    user_id bigserial REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    vector double precision[]
);

CREATE TABLE best_vectors (
    user_id bigserial REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE PRIMARY KEY,
    vector_id bigserial REFERENCES vectors(id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE id_users_bot (
    id bigserial NOT NULL PRIMARY KEY,
    user_tg_id integer NOT NULL
);