import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow.sklearn
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(data):
    """Preprocess the data using StandardScaler and OneHotEncoder."""
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month

    X = data[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
    y = data['Reading']

    categorical_cols = ['Machine_ID', 'Sensor_ID']
    numeric_cols = ['Hour', 'Day', 'Month']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

def load_latest_model():
    """Load the latest model from MLflow."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    runs = mlflow.search_runs(order_by=["start_time desc"], filter_string="")

    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/best_model")
        return model, run_id
    else:
        raise ValueError("No runs found in MLflow.")

def load_latest_data(file_path):
    """Load the latest data from the specified file path."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def calculate_metrics(model, X, y):
    """Calculate mean absolute error and mean squared error for the model."""
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    return mae, mse

def save_metrics_to_csv(metrics, csv_file):
    """Save metrics to a CSV file."""
    if not os.path.exists(csv_file):
        metrics.to_csv(csv_file, index=False)
    else:
        metrics.to_csv(csv_file, mode='a', header=False, index=False)

def get_best_mae_from_mlflow(run_id):
    """Retrieve the 'best_mae' metric from MLflow."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()

    # Fetch the metric from the specified run
    metrics = client.get_run(run_id).data.metrics
    best_mae = metrics.get("best_mae", None)

    return best_mae

def monitor(model, X, y, threshold_mae=1.0, threshold_mse=1.0):
    """Monitor model performance and print if degraded."""
    mae, mse = calculate_metrics(model, X, y)
    
    metrics_dict = {'MAE': [mae], 'MSE': [mse]}
    metrics_df = pd.DataFrame(metrics_dict)
    
    print("\nCurrent Metrics:")
    print(metrics_df)

    # Retrieve the 'best_mae' metric from MLflow
    _, run_id = load_latest_model()
    best_mae_from_mlflow = get_best_mae_from_mlflow(run_id)

    print(f"\nBest MAE from MLflow: {best_mae_from_mlflow}")

    if mae > threshold_mae or mse > threshold_mse or (best_mae_from_mlflow is not None and mae > best_mae_from_mlflow):
        print("\nModel performance degraded!")
    else:
        print("\nModel performance is within acceptable limits.")

    return metrics_df

if __name__ == "__main__":
    # Load the latest model from MLflow
    model, _ = load_latest_model()

    # Load the latest data
    data_file_path = 'dummy_sensor_data.csv'
    data = load_latest_data(data_file_path)

    # Assume data preprocessing steps are similar to model training
    X, y = preprocess_data(data)

    # Monitor model performance
    metrics_df = monitor(model, X, y)

    # Save metrics to CSV
    metrics_csv_file = 'metrics.csv'
    save_metrics_to_csv(metrics_df, metrics_csv_file)
