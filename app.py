from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model
with open("models/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    parking = int(request.form["parking"])
    house_age = int(request.form["house_age"])
    location_rating = float(request.form["location_rating"])
    distance_city = float(request.form["distance_city"])

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

    return render_template(
        "result.html",
        prediction=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)