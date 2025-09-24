# airflow/dags/dynamic_training_dag.py

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime
import pandas as pd
import mlflow
import os

def find_champion_and_build_command():
    #mlflow.set_tracking_uri("http://host.docker.internal:5000") # Docker 컨테이너가 Host PC의 MLflow 서버를 찾기 위한 주소
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()

    # 1. 태그 대신, 모델 레지스트리의 '@champion' 별칭으로 챔피언 버전을 가져옵니다.
    try:
        champion_version = client.get_model_version_by_alias("champion_model", "champion")
        print(f"현재 챔피언 모델 버전: {champion_version.version}")
    except mlflow.exceptions.MlflowException:
        raise ValueError("'@champion' 별칭이 붙은 모델을 모델 레지스트리에서 찾을 수 없습니다.")

    # 2. 해당 버전의 모델을 학습시켰던 원래의 Run 정보를 가져옵니다.
    champion_run = mlflow.get_run(champion_version.run_id)
    
    champion_params_series = champion_run.data.params

    all_possible_params = [
        "model_name", "n_estimators", "learning_rate", "max_depth", "num_leaves", 
        "min_child_samples", "subsample", "colsample_bytree", "gamma", "depth", 
        "l2_leaf_reg", "min_samples_split", "min_samples_leaf", "contamination", 
        "max_samples", "max_features"
    ]

    params_for_command = {}
    for param_key in all_possible_params:
        # params 접근 방식이 약간 달라집니다.
        if param_key in champion_params_series:
            params_for_command[param_key] = champion_params_series[param_key]

    base_command = "python /opt/airflow/dags/src/train.py "
    for key, value in params_for_command.items():
        base_command += f"--{key} {value} "

    print("생성된 최종 Bash 명령어:", base_command)
    return base_command

with DAG(
    dag_id='dynamic_fraud_training_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule='0 0,12 * * *',
    catchup=False,
    tags=['mlops', 'dynamic', 'final'],
) as dag:
    build_training_command_task = PythonOperator(
        task_id='build_training_command',
        python_callable=find_champion_and_build_command,
    )

    train_model_task = BashOperator(
        task_id='train_champion_model',
        bash_command="{{ ti.xcom_pull(task_ids='build_training_command') }}",
    )
    # 2. 학습이 끝난 후, 챔피언 모델로 배치 예측을 수행하는 새로운 태스크
    run_batch_prediction_task = BashOperator(
        task_id='run_batch_prediction',
        # load_champion_model.py 스크립트를 실행하도록 명령
        bash_command="python /opt/airflow/dags/src/load_champion_model.py"
    )

    build_training_command_task >> train_model_task >> run_batch_prediction_task