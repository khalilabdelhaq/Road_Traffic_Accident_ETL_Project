from datetime import datetime

ROOT_OUTPUT_DIR = "/opt/airflow/data/accidents_raw"
CLEAN_DIR_BASE = "/opt/airflow/data/accidents_clean"
POSTGRES_CONN_STRING = "postgresql://airflow:airflow@postgres:5432/datawarehouse"

API_URLS_BY_YEAR = {
    "2022": {
        "caracteristiques": "https://tabular-api.data.gouv.fr/api/resources/5fc299c0-4598-4c29-b74c-6a67b0cc27e7/data/",
        "lieux": "https://tabular-api.data.gouv.fr/api/resources/a6ef711a-1f03-44cb-921a-0ce8ec975995/data/",
        "vehicules": "https://tabular-api.data.gouv.fr/api/resources/c9742921-4427-41e5-81bc-f13af8bc31a0/data/",
        "usagers": "https://tabular-api.data.gouv.fr/api/resources/62c20524-d442-46f5-bfd8-982c59763ec8/data/",
    },
    "2023": {
        "caracteristiques": "https://tabular-api.data.gouv.fr/api/resources/104dbb32-704f-4e99-a71e-43563cb604f2/data/",
        "lieux": "https://tabular-api.data.gouv.fr/api/resources/8bef19bf-a5e4-46b3-b5f9-a145da4686bc/data/",
        "vehicules": "https://tabular-api.data.gouv.fr/api/resources/146a42f5-19f0-4b3e-a887-5cd8fbef057b/data/",
        "usagers": "https://tabular-api.data.gouv.fr/api/resources/68848e2a-28dd-4efc-9d5f-d512f7dbe66f/data/",
    },
}

DEFAULT_ARGS = {"start_date": datetime(2025, 8, 1), "retries": 1}
