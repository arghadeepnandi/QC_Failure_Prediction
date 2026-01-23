# QC Failure Prediction
Quality Control | Python | Flask | Machine Learning | Hugging Face

## 🎯 Predict Quality Control Failures with Machine Learning
[Live Demo](#) • [Report Bug](#) • [Request Feature](#)

---

## 📋 Table of Contents
- [About The Project](#about-the-project)
- [Features](#features)
- [Built With](#built-with)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🎯 About The Project

A **QC Failure Prediction System** powered by Machine Learning that predicts the likelihood of quality control failures in manufacturing processes. This project analyzes historical production data to provide real-time failure probability predictions, helping manufacturers prevent defects before they occur.

### Key Prediction Factors:
- 🏭 **Production Line** - Manufacturing line identifier
- 📦 **Product Type** - Type of product being manufactured
- ⚙️ **Machine Settings** - Equipment parameters and configurations
- 🌡️ **Environmental Conditions** - Temperature, humidity, pressure
- 👷 **Operator Details** - Shift information and operator experience
- ⏱️ **Production Time** - Time of day and production duration
- 📊 **Process Parameters** - Speed, pressure, temperature readings

The model analyzes critical quality metrics to predict potential failures with high accuracy, enabling proactive quality control measures.

---

## ✨ Features

- ⚡ **Real-time Predictions** - Get instant failure probability for ongoing production
- 🎯 **Multi-factor Analysis** - Considers machine, environmental, and process variables
- 🌐 **Web Interface** - Clean and intuitive Flask-based web application
- 📊 **Probability Display** - Shows pass/fail percentage for quality assessment
- 🔄 **Live Updates** - Predictions update based on current production parameters
- 🏭 **Multi-line Support** - Supports multiple production lines and configurations
- 📈 **Historical Tracking** - Monitor trends and patterns in quality metrics
- 🚨 **Alert System** - Warnings when failure probability exceeds threshold

---

## 🛠️ Built With

### Core Technologies
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python** - Primary programming language
- ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) **Flask** - Web application framework
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas** - Data manipulation and analysis
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn** - Machine learning algorithms
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) **NumPy** - Numerical computations

### Machine Learning Components
- **OneHotEncoder** - Categorical feature encoding
- **StandardScaler** - Feature normalization
- **ColumnTransformer** - Feature preprocessing pipeline
- **Pickle** - Model serialization

### Deployment
- **Hugging Face** - Model deployment platform

---

## 🌐 Live Application

Try out the live prediction model here:

👉 **[QC Failure Predictor](#)**

### Supported Production Lines
- Assembly Line A
- Assembly Line B
- Packaging Line 1
- Packaging Line 2
- Testing Station 1
- Testing Station 2

### Supported Product Types
- Product A
- Product B
- Product C
- Custom Configurations

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/QC_Failure_Prediction.git
   cd QC_Failure_Prediction
   ```

2. **Create a virtual environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file exists**  
   Make sure `qc_model.pkl` (trained model) is in the project root directory

5. **Run the Flask application**
   ```bash
   python app.py
   ```

6. **Open your browser**  
   Navigate to `http://127.0.0.1:5000`

---

## 📖 Usage

### Using the Web Interface

1. Visit the application at `http://127.0.0.1:5000` (local) or the live demo
2. **Select Production Line** - Choose the manufacturing line
3. **Select Product Type** - Pick the product being manufactured
4. **Enter Machine Settings** - Input equipment parameters
5. **Enter Environmental Conditions** - Temperature, humidity, etc.
6. **Enter Process Parameters** - Speed, pressure, readings
7. **Select Shift/Operator** - Shift information
8. **Click Predict** to get results

### Example Input
```
Production Line: Assembly Line A
Product Type: Product A
Temperature: 25°C
Humidity: 60%
Pressure: 1.2 bar
Speed: 120 units/min
Operator Shift: Morning
Experience: 5 years
```

### Example Output
```
✅ Pass Probability: 92.34%
❌ Fail Probability: 7.66%

Status: SAFE - Production within acceptable parameters
```

---

## 🧠 Model Details

### Algorithm & Architecture
- **Model Type**: Scikit-learn Pipeline with Logistic Regression/Random Forest/Gradient Boosting
- **Preprocessing**:
  - OneHotEncoder for categorical features
  - StandardScaler for numerical features
  - ColumnTransformer for unified preprocessing

### Features Used

#### Categorical Features:
- `production_line` - Manufacturing line identifier
- `product_type` - Type of product
- `shift` - Work shift (Morning/Evening/Night)
- `operator_id` - Operator identifier

#### Numerical Features:
- `temperature` - Operating temperature (°C)
- `humidity` - Humidity percentage (%)
- `pressure` - Operating pressure (bar)
- `speed` - Production speed (units/min)
- `vibration` - Machine vibration levels
- `power_consumption` - Energy usage (kW)
- `cycle_time` - Production cycle duration (seconds)
- `operator_experience` - Years of experience

### Feature Engineering

The model automatically calculates derived features:
- Temperature deviation from optimal
- Pressure-speed ratio
- Environmental stability index
- Operator efficiency score

### Model Training Pipeline
```
Raw Data → Feature Engineering → Preprocessing (OneHot + Scaling) → Model Training → Validation → Pickle Serialization
```

---

## 📊 Dataset

The model is trained on comprehensive manufacturing QC data including:

- 📅 **Historical Records**: Multiple production cycles and batches
- 🏭 **Production Scenarios**: Various machine configurations and conditions
- 🔧 **Equipment Data**: Different production lines and machinery
- 👥 **Operator Performance**: Various skill levels and shifts
- 📈 **Quality Metrics**: Pass/fail outcomes and defect classifications

---

## 📁 Project Structure

```
QC_Failure_Prediction/
│
├── app.py                      # Flask application
├── qc_model.pkl               # Trained ML model (pickle file)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── templates/
│   └── index.html            # Web interface template
│
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript files
│   └── images/               # Images and icons
│
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed datasets
│
└── notebooks/                # Jupyter notebooks (optional)
    └── model_training.ipynb
```

---

## 🔧 Dependencies

Create a `requirements.txt` file with:

```
Flask==2.3.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
pickle5==0.0.12
matplotlib==3.7.0
seaborn==0.12.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn and create. Any contributions you make are **greatly appreciated**!

### How to Contribute:

1. Fork the Project
2. Create your Feature Branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the Branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

### Ideas for Contribution:
- 🎨 Improve UI/UX design
- 📊 Add data visualizations and dashboards
- 🔄 Update production line configurations
- 🧪 Add unit tests and integration tests
- 📈 Improve model accuracy with advanced algorithms
- 📝 Enhance documentation
- 🚨 Add real-time alerting system
- 📱 Mobile-responsive design

---

## 📞 Contact

**Your Name** - Arghadeep Nandi

**Project Link**: https://github.com/arghadeepnandi/QC_Failure_Prediction (#)

**Live Demo**: [https://huggingface.co/spaces/YourUsername/qc-failure-prediction](#)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments
- [Hugging Face Spaces](https://huggingface.co/spaces) - Deployment platform


---

## 🚀 Future Enhancements

- [ ] Add real-time sensor integration via IoT
- [ ] Include predictive maintenance alerts
- [ ] API integration for ERP systems
- [ ] Advanced anomaly detection algorithms
- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Export reports to PDF/Excel
- [ ] Dashboard with historical analytics
- [ ] Integration with SCADA systems
- [ ] Batch prediction capabilities

---

⭐ **Don't forget to star this repo if you found it helpful!** ⭐

Made with ❤️ and 🏭 by **Arghadeep Nandi**

