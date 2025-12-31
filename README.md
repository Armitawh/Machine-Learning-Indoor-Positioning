# Machine Learning–Based Indoor Localization Using WiFi RSSI

## 📌 Overview
This project implements a **machine learning–driven indoor localization system** using **WiFi RSSI fingerprinting**. The objective is to accurately predict **building ID, floor level, and indoor position (longitude and latitude)** in GPS-denied environments.

Multiple supervised machine learning models are implemented and compared using a structured evaluation framework.

---

## 🧠 Implemented Machine Learning Models
The following models are implemented in MATLAB and evaluated on RSSI fingerprint datasets:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest (Bagged Trees)**
- **Linear Regression**
- **Ridge Regression**

Each model is evaluated for both **classification** (building & floor) and **regression** (position estimation).

---


## 📊 Dataset Description
- WiFi RSSI fingerprints collected from multiple access points
- Features: RSSI signal strengths
- Labels:
  - Building ID
  - Floor number
  - Longitude
  - Latitude

The dataset structure follows standard **fingerprint-based indoor localization** benchmarks (e.g., UJIIndoorLoc-style data).

---

## 📈 Evaluation Metrics

### Classification
- Building prediction accuracy
- Floor prediction accuracy

### Regression
- Mean positioning error
- Median positioning error
- Standard deviation
- Minimum and maximum error
- Error quartiles (25%, 50%, 75%)

Positioning error is calculated using **Euclidean distance** between predicted and true coordinates.

---

## 📉 Visualizations
The project generates multiple visual outputs, including:

- True vs. predicted coordinate scatter plots
- Confusion matrices
- Error histograms
- Decision tree visualizations

All figures are automatically saved for analysis.

---

## 🛠️ Tools & Technologies
- **MATLAB**
- **Machine Learning Toolbox**
- **Supervised Learning**
- **WiFi RSSI Fingerprinting**
- **Indoor Localization Systems**

---

## 🚀 Key Results
- KNN achieved the **highest localization accuracy** after tuning
- Random Forest showed strong performance in both regression and classification tasks
- Comparative analysis highlights the strengths and limitations of each ML model

---

## 🔮 Future Work
- Deep learning models (CNN / SCNN)
- Real-time RSSI input from embedded devices (e.g., Arduino, mobile sensors)
- AR-based indoor navigation visualization
- Sensor fusion (WiFi + IMU)


