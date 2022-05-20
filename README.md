# Data Engineering and Machine Learning with Airflow, Pandas, SARIMAX and XGBoost

## Purpose
Build a full data pipeline that extracts Bitcoin values in a five years interval, that makes value predictions using Machine Learning in a 30 days interval, and returns as a final product:  
- An HTML interactive graph with predictions from SARIMAX model (predictions model for univariate time series)
- An HTML interactive graph with predictions from XGBoostRegressor model (Machine Learning model for regression)
- A table with evaluation metrics from SARIMAX and XGBoostRegressor models (MSE, MAE, RMSE, MAPE, R2)  
These three files are in the folder $HOME/BitcoinPredictions after the DAG be runned successfully.  

In this project, I used XCom from Airflow to transfer variables between functions inside the BitcoinPredictions class, for what the DAG could work well and could be well structured. For this to work, I changed the following setting from 'airflow.cfg' file inside '~/airflow/dags' folder:  
> enable_xcom_pickling = True
  
The final version is in the 'bitcoin_pandas_dag.py' file.   

## Resources
- Visual Studio Code
- python3.9
- virtualenv
- pip3: python3.x packages manager

## Python packages
- airflow
- os
- pandas
- datetime
- calendar
- numpy
- plotly
- xgboost
- sklearn (Scikit-Learn)
- pmdarima

## Images from the project
<img src="image01.png" />
<img src="image02.png" />
<img src="image03.png" />


## Running this repo in your local machine
- clone this repo:  
> git clone https://github.com/rafaelcoelho1409/DataEngineering2.git  
- install required packages that are in 'dataeng2_requirements.txt' file:  
> pip3 install -r dataeng2_requirements.txt  
- choose your Python interpreter (python3.x)  
- Install Airflow:  
> pip3 install apache-airflow  
- Find the Airflow DAGs folder in your machine (generally, $AIRFLOW_HOME/dags or ~/airflow/dags)    
- copy and paste the 'bitcoin_pandas_dag.py' file into this folder ($AIRFLOW_HOME/dags or ~/airflow/dags)  
- Access Airflow from your browser (https://localhost:8080)  
- Activate the DAG (bitcoin_dag_pandas) and trigger it.  
- More orientations about Airflow data automation:
> https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html
