import joblib
import pandas as pd
import numpy as np

# Reload the Model

final_model_reloaded = joblib.load('House_Price.pkl')

# User interface

def predict_price(longitude, latitude, housing_median_age,
                  total_rooms, total_bedrooms, population,
                  households, median_income, ocean_proximity):

    valid_categories = ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]

    if ocean_proximity not in valid_categories:
        ocean_proximity = "INLAND"

    # prediction logic
    data = np.array([
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
        population, households, median_income, ocean_proximity]).reshape(1, -1)

    clms = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
        'population', 'households', 'median_income', 'ocean_proximity']

    df = pd.DataFrame(data, columns=clms)

    # prediction
    result = final_model_reloaded.predict(df)

    return result[0]