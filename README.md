# Smart Energy Load Balancer

> This repository focuses on the ML Prediction Engine trained to generate a kWh prediction based on weather conditions (temperature, humidity, cloud_cover) and school activity (time_of_day, school_factor). This prediction can then be used for a variety of tasks, but is currently used as a baseline energy consumption metric for a college campus to reschedule classes and balance energy load.

Models Trained & Tested: Linear Regression, Random Forest, Gradient Boosting, XGBoost, Ensemble

Input: Real Weather Data and Synthesized kWh data

Output: predicted_kWh

See .ipynb file for original notebook formatting for generation of .pkl files. Also included is a .py file for testing and reproducibility.

Note: fix_notebook.py only present for debugging purposes
