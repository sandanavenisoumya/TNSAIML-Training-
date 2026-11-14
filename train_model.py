import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
data = pd.read_csv("data/house_price_dataset.csv")

# Features and Target
features = [
    'Area_sqft',
    'Bedrooms',
    'Bathrooms',
    'Parking',
    'House_Age',
    'Location_Rating',
    'Distance_City_km'
]

X = data[features]
y = data['Price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print("R2 Score :", round(r2, 4))
print("MAE      :", round(mae, 2))

# Save Model
with open("models/house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully.")