# 🏠 House Price Prediction - A Regression Analysis Project

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/indeedlove/house-price-predictor)
[![GitHub](https://img.shields.io/badge/📂-View%20on%20GitHub-black)](https://github.com/inneedloveBu/house-price-prediction)
[![bilibili](https://img.shields.io/badge/🎥-View%20on%20Bilibili-yellow)](https://www.bilibili.com/video/BV1c6kJYiEua/?share_source=copy_web&vd_source=56cdc7ef44ed1ee2c9b9515febf8e9ce&t=60)

<img width="1166" height="734" alt="ScreenShot_2025-12-23_153554_201" src="https://github.com/user-attachments/assets/4f7c15a1-bba6-4bf2-86f1-e90b4c6cfd16" />



## Project Overview
This project tackles the **Kaggle House Prices** competition, a classic **regression problem**. It predicts the final sale price of residential homes in Ames, Iowa, based on over 80 explanatory features. The project showcases a complete data science pipeline, from **exploratory data analysis (EDA)** and **advanced feature engineering** to **model training with XGBoost** and **deployment as an interactive web application**.

## Key Features & Technical Highlights
*   **Complex Data Wrangling**: Handled extensive missing values by distinguishing between "true missing" and "meaning None" using domain knowledge from the data description.
*   **Advanced Feature Engineering**: Created over 10 new predictive features (e.g., `TotalSF`, `TotalBath`, `HouseAge`, `OverallGrade`) from raw variables.
*   **State-of-the-Art Modeling**: Utilized **XGBoost Regressor** and performed hyperparameter tuning via **GridSearchCV**.
*   **End-to-End Deployment**: Built and deployed an interactive web interface using **Gradio** on **Hugging Face Spaces**.

## Model Performance
After feature engineering and hyperparameter optimization, the final **XGBoost model** achieved the following performance on a held-out validation set:
*   **Root Mean Squared Error (RMSE)**: **$26,431.88**
*   **R² Score**: **0.9089**

> *Interpretation: The model can explain approximately 91% of the variance in house prices, with an average prediction error of around $26k.*

## Project Structure
<img width="640" height="200" alt="ScreenShot_2025-12-23_154646_921" src="https://github.com/user-attachments/assets/d7b7bae0-ed1f-44d6-a5df-cbcd88465980" />




## How to Run
1.  Clone the repo: `git clone https://github.com/inneedloveBu/house-price-prediction.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Download the data from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the project root.
4.  Run the training pipeline: `python model_training.py`
5.  Launch the web app locally: `python app.py`

## Live Demo
Try the interactive predictor here: [**🚀 Live Demo on Hugging Face**](https://huggingface.co/spaces/indeedlove/house-price-predictor)

## Future Improvements
*   Experiment with advanced feature selection techniques.
*   Incorporate ensemble methods (e.g., stacking with linear models).
*   Perform detailed model interpretation using SHAP values.

## License
This project is licensed under the MIT License.
