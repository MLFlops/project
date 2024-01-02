import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def feature_engineering(data):
    """Perform feature engineering on the data."""
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    return data[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']], data['Reading']

def preprocess_data(X, y):
    """Preprocess the data using StandardScaler and OneHotEncoder."""
    categorical_cols = ['Machine_ID', 'Sensor_ID']
    numeric_cols = ['Hour', 'Day', 'Month']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, y, test_size=0.2, random_state=42)

def train_model(X_train, X_val, y_train, y_val, n_estimators=100, max_depth=10):
    """Train a Random Forest Regressor model."""
    model = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mse = mean_absolute_error(y_val, predictions)
    
    return model

def grid_search(model, param_grid, X_train, y_train, X_val, y_val):
    """Perform Grid Search for hyperparameter tuning."""
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=2, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_, mean_absolute_error(y_val, grid_search.best_estimator_.predict(X_val))

def log_mlflow(run_name, params, best_model, best_mae, X_val, y_val):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_metric("best_mae", best_mae)

        run_id = run.info.run_id
        print(f"MLflow run with ID: {run_id}")

if __name__ == "__main__":
    # Step 1: Load data
    file_path = 'data/dummy_sensor_data.csv'  
    data = load_data(file_path)

    # Step 2: Feature Engineering
    X, y = feature_engineering(data)

    # Step 3: Preprocess data
    X_train, X_val, y_train, y_val = preprocess_data(X, y)

    # Step 4: Train model with optimized hyperparameters
    best_params = {'n_estimators': 100, 'max_depth': 5}  # You can replace this with the best_params obtained from grid search
    trained_model = train_model(X_train, X_val, y_train, y_val, **best_params)

    # Step 5: Grid Search for hyperparameter tuning
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    best_params, best_model, best_mae = grid_search(trained_model, param_grid, X_train, y_train, X_val, y_val)

    # Step 6: Log to MLflow
    run_name = "RandomForestRegressionRun"
    log_mlflow(run_name, best_params, best_model, best_mae, X_val, y_val)
