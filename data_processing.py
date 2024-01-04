import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Loading the dataset
data = pd.read_csv('dummy_sensor_data.csv')

# Scaling the 'Reading' feature to  to a range between 0 and 1.
scaler = MinMaxScaler()
data['Reading'] = scaler.fit_transform(data['Reading'].values.reshape(-1, 1))

# Spliting the data into training and validation sets (80% training, 20% validation)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Saving preprocessed data into separate files for training and validation
train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
