import streamlit as st
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# --- Docker 네트워크 환경을 위한 MLflow 서버 주소 설정 ---
# Streamlit 컨테이너가 'mlflow'라는 이름의 다른 컨테이너를 찾아갈 수 있도록 설정합니다.
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="신용카드 사기 탐지 모델",
    page_icon="💳",
    layout="wide"
)

# --- 앱 제목 및 설명 ---
st.title("신용카드 사기 실시간 탐지 데모 💳")
st.write("MLOps 파이프라인을 통해 배포된 모델 중 가장 좋은 성능을 가진 모델을 사용하여 실시간으로 예측합니다. (가장 좋은 모델이 변경되면 그에 맞게 업데이트 됩니다.)")
st.write("본 데모는 SMOTE와 RobustScaler를 적용한 데모용 데이터 300여개를 대상으로합니다.")
st.write("---")

# --- MLflow 모델 로딩 ---
# --- MLflow 모델 로딩 ---
# @st.cache_resource: 모델처럼 무거운 객체를 캐시에 저장하여, 앱 성능을 향상시킵니다.
@st.cache_resource
def load_production_model():
    """
    MLflow Model Registry에서 'champion' 별칭(Alias)이 붙은 모델을 불러옵니다.
    """
    model_name = "champion_model" # MLflow에 등록한 모델 이름
    model_alias = "champion"      # 사용할 별칭(Alias)
    
    # ▼▼▼ (수정) model_stage를 model_alias로 변경 ▼▼▼
    st.info(f"MLflow Registry에서 {model_name} 모델의 @{model_alias} 버전을 불러옵니다...")
    try:
        # 모델 URI는 @{alias} 형태로 특정 별명 하나를 지목합니다.
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        st.success(f"현재 사용되고 있는 모델은 \" lgbm \"입니다.\n")
        st.success("평가지표는 F1-Score = 0.78899,  AUPRC = 0.87210,  recall = 0.87755입니다.")
        return model
    except Exception as e:
        st.error(f"모델을 불러오는 데 실패했습니다: {e}")
        st.error("MLflow 서버가 실행 중인지, 모델이 '@champion' 별칭으로 등록되었는지 확인하세요.")
        return None

model = load_production_model()

# --- 데모용 테스트 데이터 로딩 ---
# @st.cache_data: 데이터처럼 변경되지 않는 객체를 캐시에 저장합니다.
@st.cache_data
def load_test_data():
    """
    데모 시연을 위해 원본 데이터에서 전처리된 테스트 데이터의 일부를 불러옵니다.
    """
    try:
        df = pd.read_csv('data/creditcard.csv')
        
        # train.py의 전처리 로직과 동일하게 스케일링을 적용합니다.
        scaler = RobustScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        # 데이터를 분리하여 테스트셋을 가져옵니다.
        _, X_test, _, y_test = train_test_split(
            df.drop('Class', axis=1), df['Class'], test_size=0.2, random_state=42, stratify=df['Class']
        )
        
        # 실제 사기 데이터와 정상 데이터를 합쳐서 데모용 데이터셋을 만듭니다.
        fraud_df = X_test[y_test == 1]
        non_fraud_df = X_test[y_test == 0].sample(200, random_state=42)
        
        demo_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        return demo_df

    except FileNotFoundError:
        st.error("'data/creditcard.csv' 파일을 찾을 수 없습니다.")
        return None

demo_data = load_test_data()

# --- 사용자 인터페이스 및 예측 ---
if model is not None and demo_data is not None:
    st.sidebar.header("⚙️ 예측 옵션")
    
    st.sidebar.info(
        "슬라이더를 움직여 테스트 데이터셋에서 예측해볼 거래를 선택하세요. "
        "실제 사기 거래와 정상 거래가 섞여 있습니다."
    )
    
    # 슬라이더로 테스트 데이터 선택
    selected_index = st.sidebar.slider("거래 선택:", 0, len(demo_data) - 1, 0)
    selected_transaction = demo_data.iloc[[selected_index]]
    
    st.subheader("📊 선택된 거래 데이터")
    st.dataframe(selected_transaction)
    
    # 예측 버튼
    if st.sidebar.button("예측 실행!", type="primary", use_container_width=True):
        with st.spinner("최신 Production 모델이 예측을 수행 중입니다..."):
            prediction = model.predict(selected_transaction)
            result = prediction[0]
            
            st.subheader("🔍 예측 결과")
            if result == 1:
                st.error("🚨 사기 의심 거래 (Fraudulent Transaction)", icon="🚨")
            else:
                st.success("✅ 정상 거래 (Normal Transaction)", icon="✅")
else:
    st.warning("모델 또는 데이터를 불러올 수 없어 데모를 실행할 수 없습니다.")

