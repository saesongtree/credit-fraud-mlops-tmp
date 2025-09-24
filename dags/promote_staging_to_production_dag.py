# dags/promote_staging_to_production_dag.py

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow

# --- 설정 변수 ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "champion_model"

def promote_staging_to_production(**context):
    """
    '@staging' 별칭이 붙은 모델을 찾아 '@champion'으로 승격시킵니다.
    """
    print("Staging -> Production(Champion) 승격을 시작합니다.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        staging_version_info = client.get_model_version_by_alias(MODEL_NAME, "staging")
        staging_version = staging_version_info.version
        print(f"승격 대상 모델을 찾았습니다: {MODEL_NAME}, 버전: {staging_version}")
        
        # '@champion' 별칭을 staging 버전에 설정합니다.
        client.set_registered_model_alias(MODEL_NAME, "champion", staging_version)
        
        # 혼동을 막기 위해 기존 '@staging' 별칭은 제거합니다.
        client.delete_registered_model_alias(MODEL_NAME, "staging")
        
        print(f"✅ 승격 완료! 새로운 챔피언은 버전 {staging_version} 입니다.")

    except mlflow.exceptions.MlflowException:
        raise ValueError("'@staging' 별칭이 붙은 모델을 찾을 수 없습니다. 먼저 Staging 모델을 지정해주세요.")

with DAG(
    dag_id='promote_staging_to_production',
    start_date=datetime(2023, 1, 1),
    schedule=None,  # <-- 수동 실행만 가능하도록 설정
    catchup=False,
    tags=['mlops', 'promotion', 'manual'],
) as dag:
    promotion_task = PythonOperator(
        task_id='promote_staging_to_production',
        python_callable=promote_staging_to_production,
    )