# Laptop Price Prediction

## Project Overview

This is my Machine Learning Zoomcamp midterm project. I built a machine learning model that estimates a laptop's price from its specifications. I explored and prepared the data in a Jupyter Notebook, compared several regression models, selected XGBoost, and saved the trained model so I could use it through an API.

I also packaged the prediction service in Docker. I used Pipenv to manage the Python environment, FastAPI to provide the prediction endpoint, and Gunicorn to run the API inside the container.

## Dataset

I used the **Laptop Price Prediction** dataset published by `arnabchaki` on Kaggle:

[https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction](https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction)

Each record includes specifications such as the manufacturer, category, screen size, screen type, CPU, RAM, storage, GPU, operating system, operating system version, and weight, along with the laptop price.

In the notebook, I downloaded the dataset with KaggleHub. The standalone training script, `train.py`, expects the dataset CSV to be available locally as `laptops.csv` in the project directory.

## Data Exploration and Preparation

I performed the data exploration and preparation in `Midterm.ipynb`. During this stage, I reviewed the dataset, checked its data types and missing values, and explored the distribution of laptop prices.

I applied the following preparation steps:

- I standardized the column names by replacing spaces with underscores and converting them to lowercase.
- I corrected the storage column name.
- I replaced missing values in `operating_system_version` with `Unknown`.
- I converted the numeric parts of RAM, weight, and screen size into numeric values when they were stored as text.
- I removed `model_name` because it has many distinct values and added unnecessary noise for this model.
- I transformed the target price using `log1p` and used `log_price` during model training. This helped reduce the effect of the long-tailed price distribution.
- I used `DictVectorizer` to convert the feature records, including categorical features, into the numerical representation required by the models.

## Model Training and Selection

In the notebook, I compared four regression approaches:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost

I first split the data into training, validation, and test sets using a 60% / 20% / 20% split. I used the validation set to compare the models and tune their hyperparameters, while keeping the test set for the final evaluation.

After comparing the models, I selected XGBoost because it achieved the best result among the configurations I tested. The selected configuration was:

| Parameter             | Value |
|---|---:|
| Learning rate (`eta`) | 0.3 |
| Maximum tree depth (`max_depth`) | 3 |
| Minimum child weight (`min_child_weight`) | 1 |
| Boosting rounds | 486 |
| Objective | `reg:squarederror` |
| Evaluation metric | `rmse` |
| Random seed | 1 |

### Test Results

These are the final test results recorded in my notebook. The RMSE values are measured on the transformed target, `log_price`.

| Model | Test Log RMSE |
|---|---:|
| Linear Regression | 0.23312 |
| Ridge Regression | 0.22972 |
| Random Forest | 0.28153 |
| **XGBoost** | **0.22417** |

XGBoost achieved the lowest test Log RMSE among the four models in this experiment.

## Training the Final Model

After completing the experiments in the notebook, I moved the final training process into `train.py` so that the model could be trained separately and saved for the API.

The script reads `laptops.csv`, applies the same main preprocessing steps, and splits the data into 80% training data and 20% test data. It then fits a `DictVectorizer`, trains XGBoost with the selected parameters, evaluates the model on the held-out test data, and saves the fitted vectorizer and trained model together in `model.bin` using Pickle.

I save both the vectorizer and the model because incoming laptop specifications must be transformed using the same fitted feature mapping that was created during training.

The 80% / 20% split in `train.py` is the final standalone training split. It is separate from the 60% / 20% / 20% training, validation, and test split used earlier in the notebook for model comparison and tuning.

## Prediction API

I used `predict.py` to serve predictions through **FastAPI**. The API loads the saved model artifact and provides the prediction endpoint below:

- **HTTP method:** POST
- **Endpoint:** `/predict`
- **Local address:** `http://localhost:5000/predict`

The endpoint receives laptop specifications as JSON and returns a price prediction. The input features include the manufacturer, category, screen size, screen type, CPU, RAM, storage, GPU, operating system, operating system version, and weight.

Because the model was trained on `log_price`, the prediction service needs to convert the model output back to the original price scale before returning the estimated price.

## API Testing

I included `test.py` to check that the prediction endpoint is responding. It sends a sample Dell Ultrabook specification to `http://localhost:5000/predict` and prints the JSON response.

The API must be running and the container must be accessible through port `5000` before I run the test. The sample in `test.py` is only an example request; users can provide other laptop specifications using the same feature names and expected data types.

## Dependencies and Environment

I used **Pipenv** to manage the project's Python dependencies and keep the environment reproducible:

- `Pipfile` lists the project's dependencies.
- `Pipfile.lock` records the resolved dependency versions.

The project uses Python libraries for data processing and model training, including Pandas, NumPy, Scikit-learn, and XGBoost. The prediction service uses FastAPI, and Gunicorn runs the application inside the Docker container.

## Docker: Build and Run

I included a `Dockerfile` to package the API and its runtime environment into a Docker image. This allows the service to run in a container without depending on the Python environment installed on the host machine.

Before running the following commands, make sure you are in the project directory that contains the `Dockerfile`.

**Windows users:** Open **Docker Desktop** first and wait until Docker has finished starting. Keep Docker Desktop running while building the image and running the container. If Docker Desktop is closed or the Docker Engine has not started, the Docker commands may fail and the container will not be created successfully.

Build the Docker image and run the container:

```bash
docker build -t laptop-price-api .
docker run -it --rm -p 5000:5000 laptop-price-api
```

Once the container is running, the API should be available at `http://localhost:5000/predict`.

The port mapping makes the API available from the local machine. It does not automatically make the service publicly accessible on the internet; public cloud deployment requires the appropriate hosting and networking configuration.

## Project Files

| File | Description |
|---|---|
| `Midterm.ipynb` | My notebook for data exploration, preprocessing, model experiments, hyperparameter tuning, and evaluation. |
| `train.py` | Trains the final XGBoost model and saves the fitted vectorizer and model in `model.bin`. |
| `predict.py` | Loads the saved model and provides the FastAPI prediction service. |
| `test.py` | Sends a sample laptop record to the prediction endpoint to test the API. |
| `Pipfile` | Defines the Pipenv environment and project dependencies. |
| `Pipfile.lock` | Records the resolved dependency versions. |
| `Dockerfile` | Defines how I build the Docker image for the prediction service. |
| `laptops.csv` | The dataset file expected by `train.py`. |
| `model.bin` | The trained model artifact generated by `train.py`. |

## Notes for Users

1. **Running the notebook:** I recommend running `Midterm.ipynb` in Google Colab to avoid many local environment and compatibility issues.
2. **Unexpected or random-looking results:** If the notebook produces unexpected results, try restarting the runtime and running all cells again from the beginning. This can happen when cells are executed out of order or when previous variables remain in memory. It does not necessarily mean that the model is making incorrect predictions.
3. **Price scale and currency:** The prices come from the dataset's local currency and can reach the millions. Therefore, a prediction may also be a large number. This can reflect the scale and distribution of the original data rather than an error in the prediction. The project does not perform currency conversion.
4. **Important Note:** When you download the project as a ZIP file, you may find a folder containing another folder (which holds the actual project files). Ensure that you work from the folder containing the project files directly—not from within another enclosing folder—as this could cause issues when building the Docker image.

## Technologies Used

- Python
- Jupyter Notebook and Google Colab
- Pandas and NumPy
- Scikit-learn
- XGBoost
- KaggleHub
- Pickle
- Pipenv
- FastAPI
- Gunicorn
- Docker
