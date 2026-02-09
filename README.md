# Patient Referral Urgency Classification - ML Analysis

This project develops a machine learning system to predict patient triage urgency levels in emergency departments. The system achieved **100% accuracy** in classifying critical patients and outperformed deep learning approaches. It is designed to support healthcare workers in resource-limited African public health facilities by providing automated, objective triage decision support.

## Quick Results

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **Test Accuracy** | 100% |
| **Critical Patient Detection** | 100% (26/26 correctly identified) |
| **Traditional ML vs Deep Learning** | ML won (100% vs 96%) |

*For detailed results, metrics, and analysis, see the full research report.*


## Dataset
Source: Kaggle (https://www.kaggle.com/datasets/hossamahmedaly/patient-priority-classification)

**Size:** 6,552 patients after data cleaning

**Features:** 18 clinical and demographic variables

- Vital signs: blood pressure, heart rate, plasma glucose, BMI
- Demographics: age, gender, residence type
- Medical history: diabetes, hypertension, heart disease
- Lifestyle: smoking status, exercise angina

**Target Variable:** Triage urgency level

- **RED** (Critical): 129 patients (2%)
- **ORANGE** (Urgent): 346 patients (5%)
- **YELLOW** (Less urgent): 5,637 patients (86%)
- **GREEN** (Minor): 440 patients (7%)

## Repository Structure
```
summative-assignment-patient-referral-urgency-ml/
├── README.md                          # Project overview 
├── requirements.txt                   # Python dependencies
├── data/
│   └── patient_priority.csv              # Dataset
├── notebook/
│   └── patient_triage_ml_analysis.ipynb # Main analysis notebook
├── src/
│   ├── model_comparison.png          # Model performance visualization
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── feature_importance.png        # Top features
│   ├── glucose_distribution.png      # Glucose by triage level
│   └── bp_age_analysis.png           # BP and age distributions
└── report/
    └── myreport.pdf              # Academic report (3,500-5,000 words)

```

### Key Achievements

-  **100% classification accuracy** using Random Forest
-  **Zero false negatives** for critical (RED) patients - all 26 critical cases correctly identified
-  **Outperformed deep learning** approaches (100% vs 96% accuracy)
-  **Clinically validated** patterns (hypotension detection, age-adjusted risk)
-  **Deployable with basic equipment** (BP cuff, glucometer, scale)


##  Methodology

### Traditional Machine Learning (Scikit-learn)
- Logistic Regression
- Decision Tree
- **Random Forest** (Best performer)
- Gradient Boosting
- K-Nearest Neighbors

### Deep Learning (TensorFlow)
- Sequential API Neural Network
- Functional API Neural Network

### Techniques Applied
- Class imbalance handling (`class_weight='balanced'`)
- Feature encoding and standardization
- Train/test split (80/20) with stratification
- Cross-validation for robustness

---

##  Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Libraries
```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
jupyter
```

### Running the Analysis

1. **Clone the repository:**
```bash
git clone https://github.com/missmbuvi04/summative-assignment-patient-referral-urgency-ml.git
cd summative-assignment-patient-referral-urgency-ml
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Open the Jupyter notebook:**
```bash
jupyter notebook notebook/patient_triage_ml_analysis.ipynb
```

4. **Run all cells** to reproduce the analysis

---

##  Presentation Video

**Video Link:** [*****************************]

**Contents:**
- Problem overview and clinical context
- Dataset description and analysis
- Methodology comparison (Traditional ML vs Deep Learning)
- Key results and findings
- Clinical implications and deployment recommendations

