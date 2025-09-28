# 💳 신용카드 사기 탐지 MLOps 파이프라인 (Credit Card Fraud Detection MLOps)

본 프로젝트는 신용카드 사기 탐지 모델을 개발하는 것을 넘어, 실험 관리, 코드 품질 검증, 모델 훈련 자동화, 그리고 최종 데모 시연에 이르는 End-to-End MLOps 워크플로우를 구축하는 것을 목표로 합니다.

## ✨ 주요 특징

* **자동화된 모델 훈련 (CT)**: `Airflow`를 사용하여 스케줄에 따라 최신 데이터로 모델을 자동으로 재학습합니다.
* **코드 품질 관리 (CI)**: `GitHub Actions`를 통해 코드 변경 시마다 자동으로 코드 스타일 검사 및 테스트를 수행합니다.
* **실험 및 모델 관리**: `MLflow`를 사용하여 모든 실험을 추적하고, 최적의 모델을 버전별로 체계적으로 관리합니다.
* **인터랙티브 데모**: `Streamlit`을 사용하여 최종 운영 모델의 성능을 실시간으로 확인할 수 있는 웹 앱을 제공합니다.
* **재현 가능한 환경**: `Docker`와 `Docker Compose`를 통해 전체 시스템을 컨테이너화하여, 어떤 환경에서든 동일한 실행을 보장합니다.

## 🏗️ 아키텍처 다이어그램

이 MLOps 파이프라인은 아래와 같은 아키텍처로 구성되어 있습니다.

 <img width="1074" height="561" alt="image" src="https://github.com/user-attachments/assets/0c6bf99e-8473-4256-8a4c-018285264e5c" />




---

## 🛠️ 기술 스택 (Tech Stack)

| 역할                  | 도구                                                         |
| :-------------------- | :----------------------------------------------------------- |
| **실험 및 모델 관리** | MLflow                                                       |
| **파이프라인 자동화 (CT)** | Apache Airflow                                               |
| **코드 품질 관리 (CI)** | GitHub Actions                                               |
| **데모 앱 개발** | Streamlit                                                    |
| **컨테이너 및 환경 관리** | Docker, Docker Compose                                       |
| **데이터 분석 및 모델링** | Pandas, Scikit-learn, LightGBM, XGBoost, etc.                |

---

## 🚀 시작하기 (Getting Started)

이 프로젝트를 로컬 환경에서 실행하기 위한 가이드입니다.

### **사전 요구사항**

* [WSL2](https://learn.microsoft.com/ko-kr/windows/wsl/install)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL2 기반으로 설정)
* Git

### **설치 및 실행**

1.  **GitHub 저장소 복제(Clone)**
    ```bash
    git clone [https://github.com/](https://github.com/)<Your-GitHub-Username>/CREDIT-FRAUD-MLOPS.git
    cd CREDIT-FRAUD-MLOPS
    ```

2.  **`.env` 파일 생성**
    프로젝트 루트에 `.env` 파일을 만들고, 아래 내용을 참고하여 암호화 키를 생성 및 추가합니다.
    ```
    # .env 파일 예시
    
    # 아래 명령어를 터미널에서 실행하여 키를 생성하고 붙여넣으세요.
    # python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    AIRFLOW__CORE__FERNET_KEY=<생성된 Fernet Key>

    # 아래 명령어를 터미널에서 실행하여 키를 생성하고 붙여넣으세요.
    # python3 -c "import secrets; print(secrets.token_hex(32))"
    AIRFLOW__API_AUTH__JWT_SECRET=<생성된 JWT Secret Key>
    ```

3.  **Docker Compose 실행**
    프로젝트 루트 폴더에서 아래 명령어를 실행하여 모든 서비스를 시작합니다.
    ```bash
    docker compose up -d --build
    ```
    최초 실행 시 이미지를 빌드하느라 시간이 다소 걸릴 수 있습니다.

4.  **서비스 접속**
    모든 서비스가 시작되면 아래 주소로 접속하여 각 UI를 확인할 수 있습니다.
    * **Airflow**: `http://localhost:8080` (ID: `airflow`, PW: `airflow` - `docker-compose.yaml`의 `airflow-init` 섹션 참조)
    * **MLflow**: `http://localhost:5000`
    * **Streamlit**: `http://localhost:8501`

---

## 📋 워크플로우 요약

이 프로젝트는 5단계의 MLOps 워크플로우를 따릅니다.

* **Phase 1: 연구 및 개발**
    * Jupyter Notebook에서 데이터를 탐색하고, `train.py`의 기본 로직을 개발합니다.

* **Phase 2: 챔피언 모델 선발**
    * `argparse`로 고도화된 `train.py`를 사용하여 다양한 모델과 파라미터로 실험을 실행하고, 모든 결과를 MLflow에 기록합니다.
    * MLflow UI에서 결과를 비교 분석하여 최적의 모델을 '챔피언'으로 선정하고 Model Registry에 등록합니다.

* **Phase 3: 모델 훈련 자동화 (Airflow)**
    * Airflow DAG가 주기적으로 MLflow에서 '챔피언' 모델의 레시피를 가져와, 최신 데이터로 모델을 자동으로 재학습시키고, 성능 검증 후 Model Registry에 새로운 '후보' 버전으로 등록합니다.

* **Phase 4: 품질 관리 및 모델 승격**
    * 운영자가 MLflow UI에서 Airflow가 등록한 '후보' 모델을 검토하고, 최종적으로 'Production' 단계로 수동 승격시킵니다.

* **Phase 5: 최종 결과 시연**
    * Streamlit 앱이 MLflow Registry에서 최신 'Production' 모델을 동적으로 불러와, 사용자에게 실시간 예측 데모를 제공합니다.

---
<img width="620" height="515" alt="image" src="https://github.com/user-attachments/assets/3eafa2a3-cef8-4777-a765-6f4c3c23fc97" />

