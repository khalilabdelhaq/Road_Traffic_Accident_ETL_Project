import os
import pandas as pd
import yaml
from sqlalchemy import create_engine
from config.settings import CLEAN_DIR_BASE, POSTGRES_CONN_STRING


def load_schema_mapping():
    with open("/opt/airflow/dags/config/schema_mapping.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve_columns(df, columns_map):
    resolved = {}
    for target, sources in columns_map.items():
        if not isinstance(sources, list):
            sources = [sources]
        for s in sources:
            if s in df.columns:
                resolved[target] = s
                break
    return resolved


def load_to_postgres(year):
    engine = create_engine(POSTGRES_CONN_STRING)
    schema_mapping = load_schema_mapping()

    tables_mapping = {
        "lieux": "dim_lieux",
        "usagers": "dim_usagers",
        "vehicules": "dim_vehicules",
        "caracteristiques": "fact_accidents",
    }

    for name, table in tables_mapping.items():
        file_path = os.path.join(CLEAN_DIR_BASE, f"{year}/{name}_clean.csv")
        df = pd.read_csv(file_path)

        if table not in schema_mapping:
            raise ValueError(f"Mapping YAML manquant pour {table}")

        columns_map = resolve_columns(df, schema_mapping[table]["columns"])
        dtypes_map = schema_mapping[table].get("dtypes", {})

        df = df.rename(columns={v: k for k, v in columns_map.items()})
        df = df[list(columns_map.keys())]

        for col, dtype in dtypes_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        if table == "fact_accidents":
            df["annee"] = int(year)

        df.to_sql(
            name=table, con=engine, if_exists="append", index=False, method="multi"
        )
        print(f"{table} ({year}) → {df.shape[0]} lignes insérées.")
