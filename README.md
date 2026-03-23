# 🏠 House Price Prediction Web App

A Machine Learning web application that predicts house prices based on user inputs such as location, income, population, and housing features.

---

## 🚀 Live Demo
🔗 https://house-price-predictor-4nlq.onrender.com/

---

## 📌 Features
- Predicts house prices in real-time
- Interactive and user-friendly web interface
- Handles both numerical and categorical inputs
- Uses a trained Machine Learning model (Random Forest)
- End-to-end ML pipeline with preprocessing + prediction
- Deployed on cloud using Render

---

## 🧠 Machine Learning Pipeline
- Data preprocessing (handling missing values, encoding, scaling)
- Feature engineering
- Model training using **RandomForestRegressor**
- Hyperparameter tuning using **RandomizedSearchCV**
- Final model saved using **joblib**

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML, CSS
- Gunicorn
- Render (Deployment)

---

## 📂 Project Structure

```
HousePricePrediction/
│
├── app.py
├── main.py
├── House_Price.pkl
├── requirements.txt
├── .gitignore
├── Procfile
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│   └── house.webp
│   └── favicon.ico
│
└── preprocessing.py
```

---

## ⚙️ How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/somyasinghal0/HousePricePrediction.git
cd HousePricePrediction
```

2. Create virtual environment:
```bash
python -m venv myenv
myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

5. Open in browser:
```
http://127.0.0.1:5000/
```

---

## 📊 Input Features
- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

## 🎯 Future Improvements
- Improve UI with Bootstrap
- Add data visualization (charts)
- Use real-world Indian housing dataset
- Add user authentication

---

## 🙌 Acknowledgements
- Scikit-learn documentation
- California Housing Dataset

---

## 📬 Contact
Feel free to connect for suggestions, feedback or collaboration.

---

⭐ If you like this project, don't forget to star the repository!
