import mlflow
import pandas as pd
import numpy as np

# --- MLflow 및 모델 정보 설정 ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "champion_model"
MODEL_ALIAS = "champion"

def load_and_predict():
    """
    MLflow 모델 레지스트리에서 '@champion' 별칭이 붙은 모델을 로드하고
    샘플 데이터로 예측을 수행합니다.
    """
    print(f"MLflow 서버에 연결을 시도합니다: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 모델 URI 형식: "models:/<모델이름>@<별칭>"
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"모델 로드를 시도합니다: {model_uri}")

    try:
        # MLflow의 PyFunc 형식으로 모델을 로드합니다. 이를 통해 원본 모델의 타입(sklearn, xgboost 등)에 관계없이
        # 통일된 방법으로 예측을 수행할 수 있습니다.
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("✅ 모델 로딩에 성공했습니다!")
    except mlflow.exceptions.MlflowException as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print(f"'{MODEL_NAME}'이라는 이름의 모델에 '@{MODEL_ALIAS}' 별칭이 설정되어 있는지 MLflow UI에서 확인해주세요.")
        return

    # --- 예측을 위한 샘플 데이터 생성 ---
    # train.py에서 사용된 피처와 동일한 구조의 더미 데이터를 생성합니다.
    # (V1-V28, scaled_amount, scaled_time) -> 총 30개 피처
    num_features = 30
    sample_data = pd.DataFrame(np.random.rand(2, num_features), 
                               columns=[f'V{i+1}' for i in range(28)] + ['scaled_amount', 'scaled_time'])
    
    print("\n--- 샘플 데이터 ---")
    print(sample_data)

    # --- 예측 수행 ---
    try:
        predictions = loaded_model.predict(sample_data)
        print("\n--- 예측 결과 ---")
        print(predictions)
    except Exception as e:
        print(f"\n❌ 예측 중 오류 발생: {e}")


if __name__ == "__main__":
    load_and_predict()