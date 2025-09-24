# dags/evaluate_and_promote_to_staging_dag.py

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow

# --- 설정 변수 ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "champion_model"
EXPERIMENT_NAME = "credit_fraud_experiment"

def get_combined_score(run):
    """주어진 Run에서 f1_score와 auprc의 합을 계산합니다."""
    f1 = run.data.metrics.get("f1_score", 0)
    auprc = run.data.metrics.get("auprc", 0)
    return f1 + auprc

def evaluate_and_promote_to_staging(**context):
    """
    최신 Run과 현재 @champion 모델을 비교하여, 성능이 더 좋으면
    '@staging' 별칭을 부여하여 검증 후보로 올립니다.
    """
    print("Staging 후보 평가 프로세스를 시작합니다.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    latest_run = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["start_time DESC"], max_results=1
    ).iloc[0]
    
    latest_run_obj = mlflow.get_run(latest_run["run_id"])
    latest_score = get_combined_score(latest_run_obj)
    
    print(f"새로운 모델(Run ID: {latest_run_obj.info.run_id}) Combined Score: {latest_score:.4f}")

    champion_score = -1.0
    try:
        champion_version_info = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = mlflow.get_run(champion_version_info.run_id)
        champion_score = get_combined_score(champion_run)
        print(f"현재 챔피언 모델(Version: {champion_version_info.version}) Combined Score: {champion_score:.4f}")
    except mlflow.exceptions.MlflowException:
        print("현재 챔피언 모델이 없습니다. 새 모델을 Staging 후보로 올립니다.")

    if latest_score > champion_score:
        print(f"🚀 새 모델이 챔피언보다 성능이 좋습니다! Staging 후보로 승격합니다.")
        model_uri = f"runs:/{latest_run_obj.info.run_id}/model"
        new_version = mlflow.register_model(model_uri, MODEL_NAME)
        
        # '@champion'이 아닌 '@staging' 별칭을 설정합니다.
        client.set_registered_model_alias(MODEL_NAME, "staging", new_version.version)
        print(f"✅ Staging 후보 등극! 모델: {MODEL_NAME}, 버전: {new_version.version}")
    else:
        print(f"⚔️ 기존 챔피언 모델의 성능이 더 좋습니다. 변경사항 없음.")

with DAG(
    dag_id='evaluate_and_promote_to_staging', # DAG ID 변경
    start_date=datetime(2023, 1, 1),
    schedule=timedelta(minutes=5),
    catchup=False,
    tags=['mlops', 'staging'],
) as dag:
    staging_task = PythonOperator(
        task_id='evaluate_and_promote_to_staging',
        python_callable=evaluate_and_promote_to_staging,
    )