from flask import Flask, render_template, request
import main
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/", methods= ["GET", "POST"])
def Home():
    result = ""

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

        result = main.predict_price(
            longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
            population, households, median_income, ocean_proximity
        )

        result = f"Your House Price is Around ₹ {result}"

    return render_template("index.html", house_value=result)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)