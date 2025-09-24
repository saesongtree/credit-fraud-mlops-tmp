# 필요한 라이브러리들을 모두 불러옵니다.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import argparse

# --- MLflow 설정 ---
MLFLOW_TRACKING_URI = "http://mlflow:5000" 
#MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"  # 필요 시 로컬 서버 주소 변경
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "credit_fraud_experiment"
mlflow.set_experiment(EXPERIMENT_NAME)



def main(args):
    """
    메인 함수: 데이터 로드, 전처리, 모델 학습, 평가 및 MLflow 로깅
    """
    print("스크립트 실행 시작...")

    # --- 1. 데이터 로드 및 전처리 ---
    try:
        df = pd.read_csv(os.path.join("/opt/airflow/data/", "creditcard.csv"))
    except FileNotFoundError:
        print("에러: 'data/creditcard.csv' 파일을 찾을 수 없습니다.")
        return

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.model_name != 'isolation_forest':
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print("데이터 전처리 완료.")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("Isolation Forest는 비지도 학습이므로 SMOTE를 적용하지 않습니다.")

    # --- 2. MLflow 실험 시작 ---
    run_name = f"{args.model_name}_run"
    with mlflow.start_run(run_name=run_name):
        print(f"MLflow Run 시작: {run_name}")
        # mlflow.autolog()
        # 파라미터 로깅
        params = {"model_name": args.model_name}

        if args.model_name == 'lgbm':
            params.update({
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_child_samples": args.min_child_samples,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree
            })
        elif args.model_name == 'xgboost':
            params.update({
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "gamma": args.gamma
            })
        elif args.model_name == 'random_forest':
            params.update({
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf
            })
        elif args.model_name == 'catboost':
            params.update({
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "depth": args.depth,
                "l2_leaf_reg": args.l2_leaf_reg
            })
        elif args.model_name == 'isolation_forest':
            params.update({
                "n_estimators": args.n_estimators,
                "contamination": args.contamination,
                "max_samples": args.max_samples,
                "max_features": args.max_features
            })

        if args.model_name != 'isolation_forest':
            params["sampling_method"] = "SMOTE"
        params["scaling_method"] = "RobustScaler"

        mlflow.log_params(params)

        # --- 3. 모델 정의 ---
        if args.model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        elif args.model_name == 'lgbm':
            model = LGBMClassifier(
                random_state=42,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                min_child_samples=args.min_child_samples,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree
            )
        elif args.model_name == 'xgboost':
            model = XGBClassifier(
                random_state=42,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                gamma=args.gamma,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif args.model_name == 'random_forest':
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth else None,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                n_jobs=-1
            )
        elif args.model_name == 'catboost':
            model = CatBoostClassifier(
                random_state=42,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                verbose=0
            )
        elif args.model_name == 'isolation_forest':
            model = IsolationForest(
                random_state=42,
                n_estimators=args.n_estimators,
                contamination=args.contamination,
                max_samples=args.max_samples,
                max_features=args.max_features,
                n_jobs=-1
            )
        else:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {args.model_name}")

        # --- 4. 모델 학습 ---
        model.fit(X_train_resampled, y_train_resampled)

        # --- 5. 모델 평가 ---
        if args.model_name == 'isolation_forest':
            preds_raw = model.predict(X_test)
            preds = [0 if p == 1 else 1 for p in preds_raw]
            scores = model.decision_function(X_test)
            preds_proba = max(scores) - scores
        else:
            preds = model.predict(X_test)
            preds_proba = model.predict_proba(X_test)[:, 1]

        auprc = average_precision_score(y_test, preds_proba)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("\n--- 모델 평가 결과 ---")
        print(f"[테스트 데이터] AUPRC: {auprc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")

        # --- 6. MLflow 메트릭 로깅 ---
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # --- 7. 모델 로깅 ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5)  # <-- 이 줄을 추가!
        )
        print("MLflow 로깅 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection MLOps Pipeline")

    parser.add_argument("--model_name", type=str, default="logistic_regression",
                        help="사용할 모델 이름: logistic_regression, lgbm, xgboost, random_forest, catboost, isolation_forest")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2_leaf_reg", type=int, default=3)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--contamination", type=float, default=0.0017)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--max_features", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
