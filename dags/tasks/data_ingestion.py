import os
import pandas as pd
import requests
from config.settings import ROOT_OUTPUT_DIR


import os
import pandas as pd
import requests
from config.settings import ROOT_OUTPUT_DIR


def fetch_and_save(name, url, year):
    output_dir = f"{ROOT_OUTPUT_DIR}/{year}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{name}.csv")

    # 🔹 Si fichier déjà présent, on skip
    if os.path.exists(output_file):
        print(f"⏩ {output_file} déjà présent, téléchargement ignoré.")
        return

    all_data = []
    page = 1
    page_size = 200  # valeur max possible selon la doc de l'API

    while True:
        paged_url = f"{url}?page={page}&page_size={page_size}"
        response = requests.get(paged_url)
        response.raise_for_status()

        json_data = response.json()
        data = json_data.get("data", [])

        if not data:
            break  # plus de données

        all_data.extend(data)
        print(f"{name} — page {page} : {len(data)} lignes récupérées")

        # 🔹 Si la page contient moins que page_size, c’est la dernière
        if len(data) < page_size:
            break

        page += 1

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"✅ {name}: {len(df)} lignes sauvegardées → {output_file}")


def preprocess_generic(file_name, year):
    input_dir = f"{ROOT_OUTPUT_DIR}/{year}"
    input_file = os.path.join(input_dir, f"{file_name}.csv")
    output_dir = f"/opt/airflow/data/accidents_clean/{year}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name}_clean.csv")

    df = pd.read_csv(input_file)
    df.to_csv(output_file, index=False)
    print(f"Prétraitement {file_name} terminé.")
