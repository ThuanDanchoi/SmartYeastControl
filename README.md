# SmartYeastControl

## Introduction

This project implements an intelligent ML-based control system for optimizing machine settings and detecting anomalies in yeast-based product manufacturing. Built for a data-driven AI assignment in manufacturing, the system leverages machine learning to analyze sensor data and recommend set points (SPs) that yield consistent product quality, while also predicting potential production downtimes.

The solution includes:

1. **Data Preprocessing**  
   Raw sensor and machine setting data is cleaned, categorized by product quality, and converted into supervised learning-ready datasets.

2. **SP Recommendation Model**  
   Models are trained to recommend optimal machine setting values (`SPs`) based on sensor readings to consistently produce high-quality output.

3. **Downtime Prediction Model**  
   Models are trained on historical shutdown data to detect whether a given sensor+SP combination could lead to failure during production.

4. **Feature Engineering and Multi-Output Regression**  
   The system supports multi-output regression where multiple SPs are predicted simultaneously using selected input features.

5. **Web-Based Visualization Interface**  
   A Streamlit-based app allows users to:
   - Upload feature inputs
   - Run SP recommendation or anomaly detection
   - View results in real-time in table format

## Project Structure
```
SmartYeastControl/
├── .venv/ # Python virtual environment
├── data/
│ ├── raw/ # Original data from factory
│ │ ├── good.csv
│ │ ├── high bad.csv
│ │ ├── low bad.csv
│ │ └── downtime.csv
│ ├── processed/ # ML-ready feature tables
│ │ ├── recommender_features.csv
│ │ └── downtime_features.csv
│ └── predictions/ # Output predictions
│ ├── recommender_predictions.csv
│ └── downtime_predictions.csv
├── models/
│ ├── recommender/ # Trained SP recommendation models
│ └── downtime/ # Trained downtime detection models
├── src/
│ ├── preprocessing/
│ │ ├── recommender.ipynb # Notebook for processing SP input data
│ │ └── downtime.ipynb # Notebook for anomaly detection data
│ ├── modeling/
│ │ ├── train_recommender.py # Train SP model
│ │ └── train_downtime.py # Train downtime model
│ ├── inference/
│ │ ├── predict_recommender.py # Inference script for SP recommendation
│ │ └── predict_downtime.py # Inference script for anomaly detection
│ └── tests/ # Optional test files
├── ui/
│ └── app.py # Streamlit web app
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Git
- Streamlit
- Virtualenv or Conda (recommended)

### Setup

```bash
git clone https://github.com/YourUsername/SmartYeastControl.git
cd SmartYeastControl
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the SP Recommender
python src/modeling/train_recommender.py

# Train the Downtime Detection Model
python src/modeling/train_downtime.py

# Run SP Prediction
python src/inference/predict_recommender.py

# Run Downtime Detection
python src/inference/predict_downtime.py

# Run Web Interface with Streamlit
streamlit run ui/app.py
```

Contributors

Duc Thuan Tran – 104330455

Dion Finnerty – 103545669

Winston Li – 104005371

Amavi Uththara – 104348238 

License
This project is developed for educational purposes as part of an AI and manufacturing assignment at Swinburne University of Technology. No official license is applied.
