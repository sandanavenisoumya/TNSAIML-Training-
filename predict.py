import pickle
import pandas as pd

# Load model
with open("models/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

print("\n===== House Price Prediction =====\n")

area = float(input("Enter Area (sqft): "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
parking = int(input("Enter Parking Spaces: "))
house_age = int(input("Enter House Age (years): "))
location_rating = float(input("Enter Location Rating (1-10): "))
distance_city = float(input("Enter Distance from City (km): "))

new_house = pd.DataFrame({
    "Area_sqft": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Parking": [parking],
    "House_Age": [house_age],
    "Location_Rating": [location_rating],
    "Distance_City_km": [distance_city]
})

prediction = model.predict(new_house)

print("\nPredicted House Price:")
print(f"₹ {prediction[0]:,.2f}")