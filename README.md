# Flight Passenger Satisfaction Prediction

This project aims to predict flight passenger satisfaction using various features related to their travel experience. The analysis involves data preprocessing, extensive exploratory data analysis (EDA), and machine learning model training and evaluation.

I created app.py to run the application using the streamlit run app.py command.

---

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Key Steps and Findings](#key-steps-and-findings)
-   [Models Used](#models-used)
-   [Best Performing Model](#best-performing-model)
-   [Contributing](#contributing)
-   [License](#license)

---

## Project Overview

The goal of this project is to build a predictive model that can determine whether a flight passenger will be "**satisfied**" or "**neutral or dissatisfied**" based on various survey responses and flight details. This can help airlines identify areas for improvement in customer service and operational efficiency.

---

## Dataset

The project utilizes a dataset named `test.csv`. This dataset contains various features related to passenger demographics, type of travel, flight information, and satisfaction ratings for different service categories.

---

## Installation

To run this notebook, you'll need to have Python installed along with the libraries listed in `requirements.txt`.

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  Ensure you have the `test.csv` file in the same directory as your Jupyter Notebook.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Flight_Passenger_Satisfaction_Prediction.ipynb"
    ```
3.  Run the cells sequentially to perform data analysis, preprocessing, model training, and evaluation.

---

## Key Steps and Findings

The notebook covers the following key steps:

1.  **Data Loading and Initial Inspection**:
    -   The dataset `test.csv` is loaded into a Pandas DataFrame.
    -   Initial inspection reveals 25 columns and a mix of numerical and categorical data types.
    -   Columns like '**Unnamed: 0**' and '**id**' are dropped as they are identifiers and not useful for analysis.

2.  **Data Preprocessing**:
    -   **Missing Values**: Missing values in the '**Arrival Delay in Minutes**' column are imputed using the mean of the column.
    -   **Invalid Ratings**: Values of `0` in rating-related columns (e.g., '**Inflight wifi service**', '**Food and drink**') are treated as missing (`NaN`) and then imputed with the mean of their respective columns, assuming a 1-5 rating scale.
    -   **Duplicates**: Checked for and confirmed no duplicate rows were present in the dataset.

3.  **Exploratory Data Analysis (EDA)**:
    -   Features are separated into numerical and categorical types.
    -   Univariate analysis of categorical features is performed using count plots and pie charts to visualize distributions (e.g., Gender, Customer Type, Type of Travel, Class, Satisfaction).
    -   Bivariate analysis investigates the relationship between categorical features and the '**satisfaction**' target variable using count plots.
    -   Univariate analysis of numerical features is performed using KDE plots to visualize distributions.
    -   Bivariate analysis examines the relationship between numerical features and '**satisfaction**' using KDE plots.
    -   Correlation analysis is performed on numerical features using a heatmap.

4.  **Feature Engineering and Encoding**:
    -   Categorical columns are converted to numerical representations using one-hot encoding for columns with multiple unique values and label encoding for binary columns. This prepares the data for machine learning models.

5.  **Model Training and Evaluation**:
    -   The dataset is split into training and testing sets.
    -   Several classification models are trained and evaluated, including:
        -   **Logistic Regression**
        -   **Decision Tree Classifier**
        -   **Random Forest Classifier**
    -   A custom function `select_best_model` is used to train and evaluate these models based on accuracy, precision, recall, and F1-score.

6.  **Model Saving**:
    -   The best-performing model is saved using `pickle` for future use.

---

## Models Used

-   **Logistic Regression**: A linear model for binary classification.
-   **Decision Tree Classifier**: A non-linear model that partitions the data based on feature values.
-   **Random Forest Classifier**: An ensemble method that builds multiple decision trees and merges their predictions.

---

## Best Performing Model

Based on the **F1-score**, the **Decision Tree Classifier** was identified as the best model for predicting flight passenger satisfaction.

---

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
