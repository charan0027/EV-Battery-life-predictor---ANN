import pandas as pd
#keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#preprocessing sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('metadata.csv')
df = df.head() 
print(df)

df = df.drop(columns=['start_time','battery_id','test_id','uid','filename'])
df.isnull().sum()
df.info()

#filling missing values
df['Re'] = pd.to_numeric(df['Re'],errors='coerce')
df['Rct'] = pd.to_numeric(df['Rct'],errors='coerce')
df['Capacity'] = pd.to_numeric(df['Capacity'],errors='coerce')

df['Re'].fillna(df['Re'].mean(),inplace=True)
df['Rct'].fillna(df['Rct'].mean(),inplace=True)
df['Capacity'].fillna(df['Capacity'].mean(),inplace=True)

df.isnull().sum()

df['type'].value_counts()
scaler = MinMaxScaler()
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])
df.head()

#Train Test Split
X = df.drop(columns=['ambient_temperature'])
y = df['ambient_temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#display data
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

#scalling 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Scaled X_train shape: \n{X_train_scaled[:5]}')

model = Sequential()
#input layer and hidden layer
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.2))

#second hidden layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

#output layer
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test))


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
# plt.show() # Commenting out to avoid blocking execution

model.evaluate(X_test_scaled, y_test)

def predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model):
    
    # Encode the categorical feature
    type_discharge_encoded = label_encoder.transform([type_discharge])[0]
    
    # Prepare the input feature vector
    X_input = np.array([[type_discharge_encoded,Capacity, Re, Rct]])
    
    # Scale the input features using the same scaler
    X_input_scaled = scaler.transform(X_input)
    
    # Predict the battery life (ambient_temperature)
    predicted_battery_life = model.predict(X_input_scaled)
    
    return predicted_battery_life[0]

# Example usage of the function
type_discharge = 'discharge'  # Example input for type
Capacity = 1.674305           # Example numeric value
Re = -4.976500e+11            # Example numeric value
Rct = 1.055903e+12            # Example numeric value

# Call the prediction function
predicted_battery_life = predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model)

print(f"Predicted Battery Life: {predicted_battery_life}")

# Example usage of the function with new input values
type_discharge = 'charge'  # New input for type
Capacity = 20.5            # Example numeric value for Capacity
Re = -2.983215e+11         # Example numeric value for Re
Rct = 1.223456e+12         # Example numeric value for Rct

# Call the prediction function with these new values
predicted_battery_life = predict_battery_life(type_discharge, Capacity, Re, Rct, label_encoder, scaler, model)

# Print the predicted battery life
print(f"Predicted Battery Life: {predicted_battery_life}")

# Save the model, scaler, and label encoder to disk
with open('battery_life_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
model.save("battery_life_model.h5")