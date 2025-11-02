-- =========================
-- Table : dim_usagers
-- =========================
CREATE TABLE dim_usagers (
    id_usager         TEXT,
    id_accident       BIGINT,
    sexe              TEXT,
    gravite           TEXT,
    categorie_usager  INTEGER,
    naissance         TEXT
);

-- =========================
-- Table : dim_vehicules
-- =========================
CREATE TABLE dim_vehicules (
    id_vehicule       TEXT,
    id_accident       BIGINT,
    type_vehicule     TEXT,
    motor             INTEGER,
    sense_circ        INTEGER
);

-- =========================
-- Table : dim_lieux
-- =========================
CREATE TABLE dim_lieux (
    id_accident       BIGINT,
    cat_route         INTEGER
);

-- =========================
-- Table : fact_accidents
-- =========================
CREATE TABLE fact_accidents (
    id_accident         BIGINT,
    jour                INTEGER,
    mois                INTEGER,
    an                  INTEGER,
    heure               TEXT,
    ville               TEXT,
    weather             INTEGER,
    luminosite          INTEGER,
    lat                 TEXT,
    long                TEXT,
    code_departement    TEXT,
    colision_type       TEXT,
    annee               INTEGER
);
