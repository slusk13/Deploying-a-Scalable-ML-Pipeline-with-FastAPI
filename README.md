# Deploying a Scalable ML Pipeline with FastAPI
### Submission Repository Link
GitHub Repository: (https://github.com/slusk13/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

## Project Overview

This project involves deploying a machine learning pipeline using FastAPI. The goal is to take a census dataset, preprocess it, train a machine learning model, and expose it via a RESTful API that performs inference. The project follows best practices for model deployment, including version control, testing, and CI/CD integration with GitHub Actions.

### Table of Contents
- [Installation Instructions](#installation-instructions)

- [Project Structure](#project-structure)

- [Data](#data)

- [Model](#model)

- [API](#api)

- [Testing](#testing)

- [CI/CD](#cicd)


## Installation Instructions
### Environment Set up (pip or conda)
1. Clone the repository:
```sh
git clone https://github.com/slusk13/Deploying-a-Scalable-ML-Pipeline-with-FastAPI
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
```
2. Install dependencies using one of the options below:
* Option 1: Use the `environment.yml` file to create a new conda environment.

```sh
conda env create -f environment.yml 
conda activate <env_name>
```
 
* Option 2: Use the `requirements.txt` file to install dependencies via pip
```sh
pip install -r requirements.txt
```
    
## Project Structure
```sh
├── data/
│   └── census.csv            # Raw dataset
├── models/
│   ├── encoder.pkl           # Saved encoder
│   └── model.pkl             # Saved model
├── ml/
│   ├── data.py               # Data processing functions
│   ├── model.py              # Model training and inference functions
│   ├── __init__.py           
├── train_model.py            # Script for training the model
├── test_ml.py                # Unit tests for the model
├── environment.yml           # Conda environment file
├── requirements.txt          # pip requirements file
└── README.md                 # Project documentation
```

## Data
The data used for this project is the `census.csv` file, which contains demographic information. The dataset is processed and used for training a machine learning model to predict income levels based on features such as age, education, occupation, etc.

- The dataset is stored in the `data/` folder and is directly loaded for use in preprocessing and model training.

## Model
The model used in this project is a classification model trained on the `census.csv` data. It predicts whether a person earns more or less than $50K annually, based on various demographic features.

- Training: The model is trained using scikit-learn on the cleaned dataset.

- Inference: Once trained, the model is saved to a .pkl file and loaded for use in the FastAPI app to make predictions.

- Performance: The model's performance can be evaluated using metrics such as accuracy and F1-score.

## API
A RESTful API is created using FastAPI to expose the trained model and provide inference capabilities.

- GET `/`: Returns a welcome message.

- POST `/data/`: Accepts JSON data (demographic features) and returns the predicted income class (either ">50K" or "<=50K").

To run the API locally:

1. Start the server:
```sh
uvicorn main:app --reload
```
2. The API will be available at http://127.0.0.1:8000

## Testing
Unit tests are included to ensure the correctness of the functions:
The tests are located in the `test_ml.py` file, which contains functional and data integrity tests using the pytest framework. Some key tests include:

- Process Data: Ensure correct data preprocessing and transformation.

- Train Model: Verify that the model is correctly trained and capable of making predictions.

- Metrics Computation: Validate that model metrics (precision, recall, F1-score) fall within the expected range.

- Data Integrity: Ensure that the age and salary columns in the dataset have valid ranges.

To run the tests:
```sh
pytest test_ml.py -v
```

## CI/CD
GitHub Actions is configured for continuous integration and deployment. The following checks are automatically run on every push:

- pytest to run the tests

- flake8 to lint the code

The configuration file for GitHub Actions is located at `.github/workflows/manual.yml`