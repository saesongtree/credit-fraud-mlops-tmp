FROM apache/airflow:3.0.6

USER root

# 시스템 패키지 업데이트 및 필요한 의존성 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Python 패키지 설치
RUN pip install --no-cache-dir \
    mlflow \
    optuna \
    numpy \
    pandas \
    scikit-learn \
    'optuna-integration[mlflow]' \
    lightgbm \
    xgboost \
    catboost \
    imbalanced-learn