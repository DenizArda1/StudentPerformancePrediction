# 🎓 Student Performance Indicator

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-cyan?style=for-the-badge&logo=githubactions)
![SHAP](https://img.shields.io/badge/XAI-SHAP-orange?style=for-the-badge)
![Validation](https://img.shields.io/badge/Validation-Pydantic-red?style=for-the-badge)

## 📖 Project Description
**Student Performance Indicator** is an end-to-end Machine Learning web application designed to predict student academic performance (Math Score) based on demographic data and academic history.

The project implements a complete Machine Learning lifecycle, including data ingestion, transformation, model training, and deployment. It is engineered with a modular architecture to ensure maintainability, utilizing Flask for the web interface, Docker for containerization, and advanced techniques like SHAP for model explainability.

---

## ⚙️ Key Technical Features

### 1. Modular Pipeline Architecture
The codebase is organized into distinct, independent components to handle specific stages of the ML lifecycle:
* **Data Ingestion:** Responsible for reading data from sources and splitting it into training and testing sets.
* **Data Transformation:** Manages preprocessing steps such as data imputation, standard scaling, and One-Hot encoding. It serializes the preprocessor object for consistent inference.
* **Model Training:** Handles model selection (Ridge Regression) and hyperparameter tuning via GridSearchCV to optimize performance.

### 2. Explainable AI (XAI) Integration
The application provides transparency for its predictions using **SHAP (SHapley Additive exPlanations)**:
* **Feature Contribution:** Calculates the specific impact of each input feature (e.g., *Reading Score, Lunch Type*) on the final prediction.
* **Dominant Factor Analysis:** The user interface dynamically displays the most influential factor affecting the predicted score.

### 3. Strict Data Validation
To ensure system stability and data integrity, the project employs **Pydantic (V2)**:
* **Schema Enforcement:** All incoming requests are validated against a defined schema before processing.
* **Type & Logic Safety:** The system automatically rejects invalid data types or out-of-range values (e.g., scores outside 0-100), preventing runtime errors in the pipeline.

### 4. Automated Retraining Mechanism
The system includes a dedicated endpoint for continuous learning:
* **`/train` Endpoint:** Triggers the full execution of the Ingestion, Transformation, and Training pipelines.
* **Seamless Updates:** Allows the model to be updated with new data in the `data/` directory without requiring a server restart or manual script execution.

### 5. Containerized Deployment
The application is fully containerized using **Docker** and **Docker Compose**:
* **Isolation:** Ensures consistent behavior across different environments (Development, Testing, Production).
* **Volume Persistence:** Maps the local `artifacts/` directory to the container, ensuring that models trained inside the Docker environment are persisted locally.

### 6. Continuous Integration (CI/CD)
The project incorporates a fully automated pipeline using **GitHub Actions**:
* **Automated Testing:** Triggers the full `pytest` suite on every push to the `main` branch to ensure code integrity.
* **Build Verification:** Verifies that the Docker image builds successfully, preventing broken deployments from reaching production.
---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Web Framework:** Flask (with Jinja2 Templates)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Explainability:** SHAP
* **Validation:** Pydantic
* **Testing:** Pytest (Unit & Integration)
* **Containerization:** Docker & Docker Compose
* **DevOps:** GitHub Actions, Docker, Docker Compose

---

## 📂 Project Structure

```text
StudentPerformancePrediction/
├── .github/workflows/      # CI/CD pipeline configurations (GitHub Actions)
├── artifacts/              # Serialized model.pkl and preprocessor.pkl (Persisted via Docker Volume)
├── data/                   # Raw CSV data source
│   └── StudentsPerformance.csv
├── src/
│   ├── components/         # Core Logic (Ingestion, Transformation, Trainer)
│   ├── pipeline/           # Orchestrators (Prediction & Training Pipelines)
│   ├── utils.py            # Utility functions
│   ├── logger.py           # Custom Logging configuration
│   ├── exception.py        # Custom Exception handling
│   └── schema.py           # Pydantic Validation Schema
├── templates/              # HTML Frontend files
├── tests/                  # Automated Test Suite
├── app.py                  # Application Entry Point
├── Dockerfile              # Docker Build Instructions
├── docker-compose.yml      # Container Orchestration
└── requirements.txt        # Project Dependencies
```

## 🚀 Getting Started

### Option 1: Run with Docker Compose (Recommended)

1.  **Clone the Repository:**
    ```bash
    git clone (https://github.com/DenizArda1/StudentPerformancePrediction.git)
    cd StudentPerformancePrediction
    ```

2.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the App:** Open `http://localhost:5000` in your browser.

> **Note:** Due to volume mapping, any model retraining performed inside the container will update the `artifacts/` folder on your local machine.

### Option 2: Run Locally (Python Virtual Env)

1.  **Create Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    python app.py
    ```

---

## 🧪 Running Tests

To verify the integrity of the pipelines, validation logic, and model behavior:

```bash
pytest
```

## 🎮 Usage Guide

## 🔌 API Endpoints

| Endpoint | Method | Description |
|--------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | Predict Math Score |
| `/train` | GET | Retrain model pipeline |


## 🎮 Usage Guide

### 1. Prediction
To use the model for prediction:
1.  Navigate to the Home Page (`http://localhost:5000`).
2.  Click **"Start Analysis"** and fill in the student details form.
3.  Click the **"Predict your Math Score"** button.
4.  The system will display the estimated Math Score and highlight the **Key Influencing Factor** (based on SHAP values).

### 2. Model Retraining
To retrain the model with new data:
1.  Update the source dataset file located at `data/StudentsPerformance.csv`.
2.  On the Home Page, click the **"Retrain Model"** button.
3.  A **loading screen** will appear while the system automatically runs the ingestion and training pipelines.
4.  Upon completion, the system will display the new **R2 Score** of the retrained model.



