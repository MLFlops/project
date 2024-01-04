from flask import Flask, render_template, request
import pandas as pd
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the pre-trained model
model_path = "D:\\mlflops\\project\\mlartifacts\\0\\dbc4cd59ca264c0285660beb7a4a66d1\\artifacts\\best_model"
loaded_model = mlflow.pyfunc.load_model(model_path)

def feature_engineering(data):
    """Perform feature engineering on the data."""
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    # Add a dummy 'Reading' column with any value (e.g., 0) for prediction
    data['Reading'] = 0
    features = data[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
    return features

def preprocess_data(X):
    """Preprocess the data using StandardScaler and OneHotEncoder."""
    categorical_cols = ['Machine_ID', 'Sensor_ID']
    numeric_cols = ['Hour', 'Day', 'Month']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed

def predict1(data):
    """Make predictions using the loaded model."""
    try:
        result = loaded_model.predict(data)
        print("Prediction result:", result)
        return result[0]  # Assuming the result is a single value
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_endpoint():
    try:
        # Get the data from the form
        timestamp = request.args.get('timestamp')
        machine_id = request.args.get('machine_id')
        sensor_id = request.args.get('sensor_id')
        
        # From timeframe, extract hour, day of week and month
        timestamp = pd.to_datetime(timestamp)
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        
        # Machine_ID_Machine_1	Machine_ID_Machine_2	Machine_ID_Machine_3	Machine_ID_Machine_4	Machine_ID_Machine_5	Sensor_ID_Sensor_1	Sensor_ID_Sensor_2	Sensor_ID_Sensor_3	Hour	DayOfWeek	Month

        # Create a DataFrame with the input values
        data = pd.DataFrame({
            'Machine_ID_Machine_1': [1 if machine_id == 'Machine_1' else 0],
            'Machine_ID_Machine_2': [1 if machine_id == 'Machine_2' else 0],
            'Machine_ID_Machine_3': [1 if machine_id == 'Machine_3' else 0],
            'Machine_ID_Machine_4': [1 if machine_id == 'Machine_4' else 0],
            'Machine_ID_Machine_5': [1 if machine_id == 'Machine_5' else 0],
            'Sensor_ID_Sensor_1': [1 if sensor_id == 'Sensor_1' else 0],
            'Sensor_ID_Sensor_2': [1 if sensor_id == 'Sensor_2' else 0],
            'Sensor_ID_Sensor_3': [1 if sensor_id == 'Sensor_3' else 0],
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'Month': [month]
        })
        

        # Make the prediction using the pre-trained model
        #prediction = predict1(X_processed)#[0]
        result = loaded_model.predict(data)[0]
        print("Prediction result:", result)
        #return result[0] 

        # Render the result template with the prediction
        return render_template('result.html', prediction=result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
