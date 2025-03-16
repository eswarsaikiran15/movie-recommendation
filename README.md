
view live demo: https://movie-recommendation-1506.streamlit.app/

# Movie Prediction using Machine Learning

## Overview

This project aims to predict movie success using Machine Learning techniques based on historical movie data stored in CSV files. The model will analyze factors like budget, genre, cast, director, and more to predict metrics such as box office revenue or IMDb ratings.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training and evaluation
- Prediction and visualization of results

## Dataset

The dataset used in this project is stored in CSV format and includes the following columns:

- **Movie Title**: Name of the movie
- **Genre**: Genre(s) of the movie
- **Director**: Director of the movie
- **Cast**: Main actors in the movie
- **Budget**: Estimated production budget
- **Revenue**: Box office revenue (target variable)
- **IMDb Rating**: User rating of the movie
- **Runtime**: Duration of the movie in minutes
- **Release Year**: Year the movie was released
- **Production Company**: Company that produced the movie

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

1. Load the CSV data using Pandas.
2. Handle missing values and data inconsistencies.
3. Convert categorical data into numerical format.
4. Normalize or scale numerical features.
5. Split data into training and testing sets.

## Exploratory Data Analysis (EDA)

- Visualizing data distribution and correlations.
- Checking feature importance.
- Detecting outliers and handling them.

## Model Training

Various machine learning models are tested and evaluated, including:

- **Linear Regression** (for predicting revenue)
- **Random Forest Regressor**
- **Gradient Boosting Models**
- **Neural Networks (optional)**

## Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-Squared Score**

## How to Run the Project

1. Clone this repository:
    
    ```bash
    git clone https://github.com/eswarsaikiran15/movie-recommendation.git
    ```
    
2. Navigate to the project folder:
    
    ```bash
    cd movie_recommendation
    ```
    
3. Run the Python script:
    
    ```bash
    python movie_recommender.py
    ```
    

## Results and Visualization

- The trained model's performance is evaluated on test data.
- Predictions are visualized using matplotlib and seaborn.

## Future Enhancements

- Improve feature engineering for better accuracy.
- Use deep learning models for enhanced predictions.
- Deploy the model as a web application.



