# **Advanced Data Science Project**  

## **Topic: Customer Churn Prediction with Automated ML Pipelines**  

---

## Setup

#### Create a Virtual Environment
```bash
conda create -p venv python==3.12.5 -y
```

```bash
conda activate venv/
```

#### Git Configuration
```bash
git config --global user.name "USER_NAME"
```

```bash
git config --global user.email "USER_EMAIL"
```
---
### **Objective**  
Develop an end-to-end, scalable machine learning system to predict customer churn and provide actionable insights. The system will use custom datasets for prototyping and advanced tools like ZenML for pipeline management and MLflow for experiment tracking. It will also include modern deployment practices with FastAPI, Docker, GitHub Actions (for CI/CD), and Heroku.

---

### **Goal**  
1. Build an end-to-end ML solution that:  
   - Identifies customers at risk of leaving a subscription-based service.  
   - Explains the key factors influencing churn using interpretable models.  
   - Provides recommendations for targeted retention strategies based on data-driven insights.  
2. Deploy the solution as a scalable and user-friendly system accessible to business stakeholders.  

---

### **Problem Statement**  
In competitive industries like telecom, e-commerce, and SaaS, retaining customers is more cost-effective than acquiring new ones. However, identifying customers at risk of churn and understanding the reasons behind it is challenging. Companies struggle with:  
1. Predicting churn accurately due to noisy and imbalanced datasets.  
2. Extracting actionable insights from churn models for targeted retention campaigns.  
3. Automating the churn prediction process while ensuring scalability for real-time predictions.  

This project aims to address these challenges by building a machine learning solution that not only predicts churn but also explains its drivers and integrates these insights into actionable business strategies.  

---

### **Dataset**  

[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

---

### **Tools and Technologies**  
- **Data Processing:** [Pandas](https://pandas.pydata.org/docs/), [Scikit-learn](https://scikit-learn.org/stable/)  
- **Pipeline Management:** [ZenML](https://docs.zenml.io/) 
- **Experiment Tracking:** [MLflow](https://mlflow.org/docs/latest/index.html)    
- **Deployment:** [FastAPI](https://fastapi.tiangolo.com/), [Docker](https://docs.docker.com/), [Heroku](https://www.heroku.com/)  
- **CI/CD:** [GitHub Actions](https://docs.github.com/en/actions)  

---

### **Deliverables**  
1. **Interactive REST API:**  
   - Predict churn probability and explain predictions.  
2. **Reproducible ML Pipeline:**  
   - ZenML-based pipeline integrated with MLflow for tracking.  
3. **Deployed Solution:**  
   - Fully operational system on Heroku with automated CI/CD.  

---

### **Model Development**  
   - **Baseline Model:** Logistic Regression.  
   - **Advanced Models:** Decision Trees, XGBoost, LightGBM, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN). 
   - **Model Evaluation:** Focus on metrics like:  
     - Precision and Recall (to minimize false negatives).  
     - F1-score and AUC-ROC for overall performance.  
   - **Hyperparameter Optimization:** Use GridSearchCV.  

---


### **Project Implementation Steps**  

#### **1. Prototyping**  
 - **Prototyping Workflow:**  
   The rapid prototyping design uses reusable custom modules and artifacts for handling data ingestion, preprocessing, and model building, where we can:  
   - Adjust configuration easily.  
   - Test different datasets quickly.  
   - Extend functionality by modifying or adding new steps to the data ingestion, transformation, and training processes without breaking the existing pipeline.

#### **2. Pipeline Implementation**  
   - **Pipeline Design with ZenML:**  
     - **Steps:**  
       1. Data Ingestion.  
       2. Data Preprocessing and Feature Engineering.  
       3. Model Training.  
       4. Evaluation and Experiment Tracking.  
     - **Reproducibility:** Use ZenML to version pipelines and ensure consistent execution.  
   - **Integration with MLflow:** Track model parameters, performance metrics, and artifacts like plots and serialized models.  

---

#### **3. Deployment with FastAPI**  
   - **API Development:**  
     - Build REST API endpoints:  
       - **/predict:** Accepts customer data and returns churn probability.  
       - **/explain:** Provides feature-based explanations using SHAP.  
     - Ensure JSON-based request/response schema for seamless integration.  

---

#### **4. Containerization with Docker**  
   - **Containerization:**  
     - Create a Dockerfile to containerize the API and pipeline.  
     - Use multi-stage builds to optimize the image size.  
   - **Testing:** Test a Docker container locally using Docker Compose.  

---

#### **5. CI/CD with GitHub Actions**  
   - **Automated Pipeline:**  
     - Run tests on every push or pull request.  
     - Build and push Docker images to a container registry (Docker Hub).  
     - Deploy to Heroku automatically after successful builds.  

---

#### **6. Deployment on Heroku**  
   - **Setup:**  
     - Deploy the Dockerized API to Heroku.  
     - Configure environment variables and storage (if needed).  
   - **Scaling:** Use Heroku’s dynos for scaling based on API usage.  

---

#### **7. Monitoring and Retraining**  
   - **Monitoring:**  
     - Track model performance using MLflow.  
     - Monitor data drift using ZenML’s built-in tools.  
   - **Retraining Pipeline:** Automate data updates and model retraining using ZenML.  

---


Architecture diagram for each of the project implementation steps for your **Customer Churn Prediction with Automated ML Pipelines**:

---

### **1. Prototyping Workflow Architecture Diagram**

```plaintext
+-------------------------------+
|        Prototype Setup         |
| (Custom Modules + Artifacts)   |
+-------------------------------+
           |
           V
+-------------------------------+
|     Data Ingestion Module      |
|    (Read Data from CSV)        |
+-------------------------------+
           |
           V
+-------------------------------+
|  Data Preprocessing (Clean)    |
|    (Handle Missing, Encoding)  |
+-------------------------------+
           |
           V
+-------------------------------+
|      Feature Engineering       |
|  (Scaling,Transformation)      |
+-------------------------------+
           |
           V
+-------------------------------+
|        Model Training          |
|   (Logistic Regression,e.t.c)  |
+-------------------------------+
           |
           V
+-------------------------------+
|     Model Evaluation & Tuning  |
| (Hyperparameter Optimization,  |
| GridSearchCV, Cross-Validation)|
+-------------------------------+
           |
           V
+-------------------------------+
|        Save Artifacts          |
|     (Model, Metrics, Visuals)  |
+-------------------------------+
```

---

### **2. Pipeline Implementation with ZenML Architecture Diagram**

```plaintext
+-------------------------------+
|      ZenML Pipeline           |
|   (Orchestrates All Steps)    |
+-------------------------------+
           |
           V
+-------------------------------+      +-------------------------------+
|    Data Ingestion Step        | ---> |  Data Preprocessing & Feature  |
| (Read and Load CSV File)      |      |     Engineering Step           |
|                               |      | (Handle Missing Data, Encoding)|
+-------------------------------+      +-------------------------------+
           |
           V
+-------------------------------+
|     Model Training Step       |
|  (Train Models: LR, XGBoost)  |
+-------------------------------+
           |
           V
+-------------------------------+
|    Evaluation & Metrics       |
|  (Track Metrics with MLflow)  |
+-------------------------------+
           |
           V
+-------------------------------+
|     Model Artifact Storage    |
|  (Store Models & Visuals in   |
|   ZenML Artifacts)            |
+-------------------------------+
           |
           V
+-------------------------------+
|     ZenML Pipeline Versioning |
|(Version Control for Pipelines)|
+-------------------------------+
```

---

### **3. Deployment with FastAPI Architecture Diagram**

```plaintext
+-------------------------------+
|       FastAPI App             |
|   (Expose RESTful Endpoints)  |
+-------------------------------+
           |
           V
+-------------------------------+      +-------------------------------+
|      /predict Endpoint        | ---> |   /explain Endpoint           |
|  (Accept Data, Predict Churn) |      | (Explain Predictions via SHAP)|
+-------------------------------+      +-------------------------------+
           |
           V
+-------------------------------+
|       Model Inference Layer   |
| (Load Pretrained Model)       |
+-------------------------------+
           |
           V
+-------------------------------+
|     Churn Prediction          |
|   (Return Prediction Result)  |
+-------------------------------+
           |
           V
+-------------------------------+
|     Explanation Layer         |
|  (Provide SHAP Explanations)  |
+-------------------------------+
```

---

### **4. Containerization with Docker Architecture Diagram**

```plaintext
+-------------------------------+
|         Docker Build           |
|   (Create Dockerfile for App)  |
+-------------------------------+
           |
           V
+-------------------------------+
|   Docker Multi-Stage Build    |
| (Optimize Image Size)         |
+-------------------------------+
           |
           V
+-------------------------------+
|     Docker Container           |
|  (FastAPI App + Model Pipeline)|
+-------------------------------+
           |
           V
+-------------------------------+
|  Docker Compose (Test Locally) |
|    (Run the Container Locally) |
+-------------------------------+
```

---

### **5. CI/CD with GitHub Actions Architecture Diagram**

```plaintext
+-------------------------------+
|      GitHub Actions Workflow  |
|   (Automated CI/CD Pipeline)  |
+-------------------------------+
           |
           V
+-------------------------------+      +-------------------------------+
|       Automated Testing       | ---> |      Build Docker Image       |
|   (Unit & Integration Tests)  |      |   (Push to Docker Hub)        |
+-------------------------------+      +-------------------------------+
           |
           V
+-------------------------------+
|    Deploy to Heroku           |
| (Push Docker Image to Heroku) |
+-------------------------------+
           |
           V
+-------------------------------+
|     Continuous Deployment     |
| (Automatically Deploy Changes)|
+-------------------------------+
```

---

### **6. Deployment on Heroku Architecture Diagram**

```plaintext
+-------------------------------+
|         Heroku Setup          |
|    (Heroku App Configuration) |
+-------------------------------+
           |
           V
+-------------------------------+
|      Docker Deployment        |
|   (Push Dockerized App to     |
|      Heroku)                  |
+-------------------------------+
           |
           V
+-------------------------------+
|       Environment Variables   |
|     (Configure Storage & API) |
+-------------------------------+
           |
           V
+-------------------------------+
|     Scaling (Heroku Dynos)    |
|  (Scale based on Traffic Load)|
+-------------------------------+
           |
           V
+-------------------------------+
|     Production API Running    |
|    (Access through Heroku URL)|
+-------------------------------+
```

---

### **7. Monitoring & Retraining Architecture Diagram**

```plaintext
+-------------------------------+
|      Monitoring with MLflow   |
|  (Track Model Performance)    |
+-------------------------------+
           |
           V
+-------------------------------+      +-------------------------------+
|    Track Data Drift with ZenML| ---> | Retraining Triggered on Drift |
| (Monitor Metrics & Data Drift)|      |  (Retrain with New Data)      |
+-------------------------------+      +-------------------------------+
           |
           V
+-------------------------------+
|       Retraining Pipeline     |
|   (Automated Model Retraining)|
+-------------------------------+
           |
           V
+-------------------------------+
|   Update Pipeline Artifacts   |
|  (New Model, Updated Metrics) |
+-------------------------------+
```

---

### **Summary:**

1. **Prototyping Workflow:** Focuses on rapid development, using custom modules for data ingestion, preprocessing, training, and saving artifacts.
2. **Pipeline Implementation:** ZenML orchestrates the entire pipeline, ensuring steps like data ingestion, preprocessing, training, evaluation, and model artifact management are versioned and reproducible.
3. **FastAPI Deployment:** Exposes endpoints for churn prediction and explanation using SHAP, ensuring an easy interface for external integration.
4. **Docker Containerization:** Ensures portability and consistency by creating a containerized environment for the FastAPI app and model pipeline.
5. **CI/CD with GitHub Actions:** Automates the testing, Docker image building, and deployment processes, ensuring continuous integration and delivery.
6. **Deployment on Heroku:** Focuses on deploying the system to the cloud with scalable infrastructure, utilizing Heroku’s services.
7. **Monitoring & Retraining:** Tracks model performance and data drift, triggering automatic retraining when necessary, ensuring the system remains accurate over time.
