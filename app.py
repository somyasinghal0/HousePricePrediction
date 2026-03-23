from flask import Flask, render_template, request
import main
from main import final_model_reloaded as model
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/", methods= ["GET", "POST"])
def Home():
    result = None
    
    if request.method == "POST":
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        housing_median_age = float(request.form.get("housing_median_age"))
        total_rooms = float(request.form.get("total_rooms"))
        total_bedrooms = float(request.form.get("total_bedrooms"))
        population = float(request.form.get("population"))
        households = float(request.form.get("households"))
        median_income = float(request.form.get("median_income"))
        ocean_proximity = request.form.get("ocean_proximity")

        # prediction logic
        data = np.array([
            longitude, latitude, housing_median_age,total_rooms, total_bedrooms, 
            population,households, median_income, ocean_proximity]).reshape(1, -1)

        clms = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
            'population', 'households', 'median_income', 'ocean_proximity']

        df = pd.DataFrame(data, columns=clms)

        result = model.predict(df)
        result = f"Your House Price is Around ₹ {result}"

    return render_template("index.html", house_value=result)

app.run(debug=True)