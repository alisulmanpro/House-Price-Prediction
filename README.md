
# 🏠 House Price Prediction System (Machine Learning + FastAPI)

## 📌 Project Overview

This project is an **end-to-end Machine Learning system** that predicts **house prices** based on neighborhood and housing features such as crime rate, number of rooms, pollution level, and socio-economic factors.

The model is trained on the **Boston Housing Dataset** and deployed using **FastAPI** to provide real-time predictions through a REST API.

This project demonstrates **practical ML engineering skills**, not just model training.

---

## 🎯 Problem Statement

House prices depend on many factors:

* Crime rate
* Number of rooms
* Location advantages (river proximity)
* Pollution level
* Tax rate
* Socio-economic status

The goal is to **learn patterns from historical data** and predict the **median house value (MEDV)** accurately.

---

## 📊 Dataset

* **Source:** Boston Housing Dataset
* **Records:** 506 houses
* **Features:** 13 numerical features
* **Target:** `MEDV` (Median value of owner-occupied homes)

### Feature Description (Simple Meaning)

| Feature | Meaning                      |
| ------- | ---------------------------- |
| CRIM    | Crime rate in area           |
| ZN      | Residential land percentage  |
| INDUS   | Industrial area proportion   |
| CHAS    | Near Charles River (1 = Yes) |
| NOX     | Air pollution level          |
| RM      | Average number of rooms      |
| AGE     | Age of houses                |
| DIS     | Distance to city centers     |
| RAD     | Road accessibility           |
| TAX     | Property tax rate            |
| PTRATIO | Student-teacher ratio        |
| B       | Population demographic score |
| LSTAT   | % of low-income population   |
| MEDV    | House price (Target)         |

---

## 🧠 Machine Learning Approach

* **Problem Type:** Regression
* **Algorithm:** Random Forest Regressor
* **Why Random Forest?**

  * Handles non-linear relationships well
  * Robust to outliers
  * Strong performance on tabular data

---

## 📈 Model Performance

| Metric                     | Value |
|----------------------------|-------|
| MAE                        | ~2.06 |
| RMSE                       | ~2.92 |
| R² Score                   | ~0.88 |

✔ Model explains ~88% of price variance <br>
✔ Average prediction error ≈ ±5 price units

This is **solid performance** for this dataset.

---

## 🗂 Project Structure

```
house-price-prediction/
│
├── data/
│   └── boston.csv
│
├── notebooks/
│   └── eda_and_training.ipynb
│
├── model/
│   ├── house_price_model.pkl
│   └── scaler.pkl
│
├── main.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 🚀 FastAPI Integration

The trained model is exposed via a **REST API** using FastAPI.

### API Features

* Accepts house features as JSON
* Returns predicted house price
* Swagger UI available for testing

### Run the API

```bash
fastapi run main.py
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 Example API Input

```json
{
  "CRIM": 0.3,
  "ZN": 12,
  "INDUS": 7.0,
  "CHAS": 0,
  "NOX": 0.47,
  "RM": 6.2,
  "AGE": 60,
  "DIS": 4.0,
  "RAD": 5,
  "TAX": 320,
  "PTRATIO": 16.5,
  "B": 380,
  "LSTAT": 14.0
}
```

### Example Output

```json
{
  "predicted_house_price": 23.84
}
```

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Random Forest
* FastAPI
* Joblib

---

## 📌 Key Learnings

* Real-world data preprocessing
* Feature importance analysis
* Regression evaluation metrics
* Model serialization
* API-based ML deployment
* Debugging ML pipelines (scaling issues)

---

## 🔮 Future Improvements

* Dockerize the application
* Deploy on cloud (Render / AWS / Railway)
* Add model versioning
* Add input validation & logging
* Try Gradient Boosting / XGBoost

---

## 👤 Author

**Ali Sulman** <br>
Aspiring Machine Learning Engineer<br>
Focused on production-ready ML systems

