from airflow import DAG
from airflow.operators.python import PythonOperator
from config.settings import DEFAULT_ARGS, API_URLS_BY_YEAR
from tasks.data_ingestion import fetch_and_save, preprocess_generic
from tasks.data_loading import load_to_postgres
import subprocess


dag = DAG(
    dag_id="download_accidents_api",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    catchup=False,
    tags=["api", "data.gouv", "download"],
)

preprocess_funcs = ["caracteristiques", "lieux", "vehicules", "usagers"]

for year, urls in API_URLS_BY_YEAR.items():
    preprocess_tasks = []
    for name, url in urls.items():
        download = PythonOperator(
            task_id=f"download_{name}_{year}",
            python_callable=fetch_and_save,
            op_args=[name, url, year],
            dag=dag,
        )
        preprocess = PythonOperator(
            task_id=f"preprocess_{name}_{year}",
            python_callable=preprocess_generic,
            op_args=[name, year],
            dag=dag,
        )
        download >> preprocess
        preprocess_tasks.append(preprocess)

    load = PythonOperator(
        task_id=f"load_to_postgres_{year}",
        python_callable=load_to_postgres,
        op_args=[year],
        dag=dag,
    )
    for p in preprocess_tasks:
        p >> load


def run_training_script():
    subprocess.run(["python", "/opt/airflow/dags/train_models.py"], check=True)


train_task = PythonOperator(
    task_id="train_models",
    python_callable=run_training_script,
    dag=dag,
)

# Lier à la fin de ton DAG
for year in API_URLS_BY_YEAR.keys():
    dag.get_task(f"load_to_postgres_{year}") >> train_task
