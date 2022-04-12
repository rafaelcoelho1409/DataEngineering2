######################### Imports ###############################
import airflow
import os
import pandas as pd
import datetime
import calendar
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from datetime import timedelta
from pmdarima import auto_arima
from airflow import DAG 
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
###################################################################

################## Airflow ########################################
args = {
    'owner': 'rafaelcoelho',
    'start_date': airflow.utils.dates.days_ago(3),
    'depends_on_past': False,
    'retries': 5,
    'retries_delay': timedelta(seconds = 30)}

dag = DAG(
    dag_id = 'bitcoin_dag_pandas',
    default_args = args,
    schedule_interval = timedelta(days = 1))
###################################################################

################################## Functions ######################
class BitcoinPredictions:
    def __init__(self):
        self.HOME = os.environ['HOME']

    def create_folder(self):
        os.chdir(self.HOME)
        try:
            os.mkdir('BitcoinPredictions')
        except:
            pass
        os.chdir('BitcoinPredictions')

    def get_data(self, **context):
        os.chdir(f'{self.HOME}/BitcoinPredictions')
        now = datetime.datetime.now()
        start_date = calendar.timegm((now.year - 5, now.month, now.day, 0, 0, 0))
        end_date = calendar.timegm((now.year, now.month, now.day, 0, 0, 0))
        self.df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.date_path = '{}/{}'.format(f'{self.HOME}/BitcoinPredictions', datetime.datetime.now())
        context['task_instance'].xcom_push(key = 'df', value = self.df)
        context['task_instance'].xcom_push(key = 'date_path', value = self.date_path)

    def sarimax_model(self, **context):
        self.data = context['task_instance'].xcom_pull(key = 'df', task_ids = ['get_data'])[0]
        self.data = self.data.dropna()
        X = self.data['Close']
        exoX = self.data['Low']
        self.sarimax_train, self.sarimax_test = X[:-30], X[-30:]
        self.sarimax_exotrain, self.sarimax_exotest = exoX[:-30], exoX[-30:]
        self.sarimax = auto_arima(
            np.array(self.sarimax_train).reshape(-1, 1),
            exogenous = np.array(self.sarimax_exotrain).reshape(-1, 1),
            start_p = 1,
            start_q = 1,
            max_p = 7,
            max_q = 7,
            seasonal = True,
            start_P = 1,
            start_Q = 1,
            max_P = 1,
            max_D = 1,
            max_Q = 7,
            d = None,
            D = None,
            trace = True,
            error_action = 'ignore',
            suppress_warnings = True,
            stepwise = True)
        sarimax_pred, sarimax_conf_int = self.sarimax.predict(
            n_periods = 30, 
            exogenous = np.array(self.sarimax_exotest).reshape(-1, 1),
            return_conf_int = True)
        self.sarimax_conf_int = pd.DataFrame(sarimax_conf_int, columns = ['Upper_bound', 'Lower_bound'])
        self.sarimax_pred = pd.DataFrame(sarimax_pred, columns = ['Predição'])
        context['task_instance'].xcom_push(key = 'sarimax_train', value = self.sarimax_train)
        context['task_instance'].xcom_push(key = 'sarimax_test', value = self.sarimax_test)
        context['task_instance'].xcom_push(key = 'sarimax_exotrain', value = self.sarimax_exotrain)
        context['task_instance'].xcom_push(key = 'sarimax_exotest', value = self.sarimax_exotest)
        context['task_instance'].xcom_push(key = 'sarimax_pred', value = self.sarimax_pred)
        context['task_instance'].xcom_push(key = 'data', value = self.data)

    def sarimax_plot(self, **context):
        self.sarimax_train = context['task_instance'].xcom_pull(key = 'sarimax_train', task_ids = ['sarimax_model'])[0]
        self.sarimax_test = context['task_instance'].xcom_pull(key = 'sarimax_test', task_ids = ['sarimax_model'])[0]
        self.sarimax_exotrain = context['task_instance'].xcom_pull(key = 'sarimax_exotrain', task_ids = ['sarimax_model'])[0]
        self.sarimax_exotest = context['task_instance'].xcom_pull(key = 'sarimax_exotest', task_ids = ['sarimax_model'])[0]
        self.sarimax_pred = context['task_instance'].xcom_pull(key = 'sarimax_pred', task_ids = ['sarimax_model'])[0]
        self.data = context['task_instance'].xcom_pull(key = 'data', task_ids = ['sarimax_model'])[0]
        self.date_path = context['task_instance'].xcom_pull(key = 'date_path', task_ids = ['get_data'])[0]
        fig1 = px.line(self.data['Close'].rename('Treino')[-60:-29], title = 'Predição de valores de Bitcoin usando SARIMAX ({} - {})'.format(
            str((datetime.datetime.now() - datetime.timedelta(days = 30)).strftime('%d/%m/%Y')),
            datetime.datetime.now().strftime('%d/%m/%Y')))
        fig1.data[0].line.color = "#0000ff"
        fig1.data[0].x = pd.to_datetime(self.data['Date'])[-60:-29]
        fig2 = px.line(self.sarimax_test.rename('Teste'))
        fig2.data[0].line.color = "#ff0000"
        fig2.data[0].x = pd.to_datetime(self.data['Date'])[-30:]
        fig1.add_trace(fig2.data[0])
        fig3 = px.line(self.sarimax_pred)
        fig3.data[0].line.color = "#ffa500"
        fig3.data[0].x = pd.to_datetime(self.data['Date'])[-30:]
        fig1.add_trace(fig3.data[0])
        fig1.update_xaxes(title_text = 'Data')
        fig1.update_yaxes(title_text = 'Valores (em US$)')
        try:
            os.mkdir(self.date_path)
        except:
            pass
        os.chdir(self.date_path)
        fig1.write_html('sarimax_predictions.html')

    def xgboost_model(self, **context):
        self.data = context['task_instance'].xcom_pull(key = 'df', task_ids = ['get_data'])[0]
        self.data = self.data.dropna()
        self.data = self.data.drop(['Date'], axis = 1)
        self.xgboost_x_train = self.data.drop(['Close'], axis = 1)[:-30]
        self.xgboost_x_test = self.data.drop(['Close'], axis = 1)[-30:]
        self.xgboost_y_train = self.data['Close'][:-30]
        self.xgboost_y_test = self.data['Close'][-30:]
        self.xgboost = XGBRegressor()
        params = {
            'max_depth': [5, 10, 15, 20, 25],
            'n_estimators': [50, 100]}
        self.x_grid_model = GridSearchCV(
            self.xgboost,
            params,
            cv = 3)
        self.x_grid_model.fit(self.xgboost_x_train, self.xgboost_y_train)
        self.xgboost_y_pred = self.x_grid_model.predict(self.xgboost_x_test)
        self.xgboost_y_pred = pd.DataFrame(self.xgboost_y_pred, columns = ['Predição'])
        context['task_instance'].xcom_push(key = 'xgboost_x_train', value = self.xgboost_x_train)
        context['task_instance'].xcom_push(key = 'xgboost_y_train', value = self.xgboost_y_train)
        context['task_instance'].xcom_push(key = 'xgboost_x_test', value = self.xgboost_x_test)
        context['task_instance'].xcom_push(key = 'xgboost_y_test', value = self.xgboost_y_test)
        context['task_instance'].xcom_push(key = 'xgboost_y_pred', value = self.xgboost_y_pred)

    def xgboost_plot(self, **context):
        self.xgboost_x_train = context['task_instance'].xcom_pull(key = 'xgboost_x_train', task_ids = ['xgboost_model'])[0]
        self.xgboost_y_train = context['task_instance'].xcom_pull(key = 'xgboost_y_train', task_ids = ['xgboost_model'])[0]
        self.xgboost_x_test = context['task_instance'].xcom_pull(key = 'xgboost_x_test', task_ids = ['xgboost_model'])[0]
        self.xgboost_y_test = context['task_instance'].xcom_pull(key = 'xgboost_y_test', task_ids = ['xgboost_model'])[0]
        self.xgboost_y_pred = context['task_instance'].xcom_pull(key = 'xgboost_y_pred', task_ids = ['xgboost_model'])[0]
        self.data = context['task_instance'].xcom_pull(key = 'data', task_ids = ['sarimax_model'])[0]
        self.date_path = context['task_instance'].xcom_pull(key = 'date_path', task_ids = ['get_data'])[0]
        fig1 = px.line(self.data['Close'].rename('Treino')[-60:-29], title = 'Predição de valores de Bitcoin usando XGBoost ({} - {})'.format(
            str((datetime.datetime.now() - datetime.timedelta(days = 30)).strftime('%d/%m/%Y')),
            datetime.datetime.now().strftime('%d/%m/%Y')))
        fig1.data[0].line.color = "#0000ff"
        fig1.data[0].x = pd.to_datetime(self.data['Date'])[-60:-29]
        fig2 = px.line(self.xgboost_y_test.rename('Teste'))
        fig2.data[0].line.color = "#ff0000"
        fig2.data[0].x = pd.to_datetime(self.data['Date'])[-30:]
        fig1.add_trace(fig2.data[0])
        fig3 = px.line(self.xgboost_y_pred)
        fig3.data[0].line.color = "#ffa500"
        fig3.data[0].x = pd.to_datetime(self.data['Date'])[-30:]
        fig1.add_trace(fig3.data[0])
        fig1.update_xaxes(title_text = 'Data')
        fig1.update_yaxes(title_text = 'Valores (em US$)')
        try:
            os.mkdir(self.date_path)
        except:
            pass
        os.chdir(self.date_path)
        fig1.write_html('xgboost_predictions.html')

    def metrics(self, **context):
        self.sarimax_test = context['task_instance'].xcom_pull(key = 'sarimax_test', task_ids = ['sarimax_model'])[0]
        self.sarimax_pred = context['task_instance'].xcom_pull(key = 'sarimax_pred', task_ids = ['sarimax_model'])[0]
        self.xgboost_y_test = context['task_instance'].xcom_pull(key = 'xgboost_y_test', task_ids = ['xgboost_model'])[0]
        self.xgboost_y_pred = context['task_instance'].xcom_pull(key = 'xgboost_y_pred', task_ids = ['xgboost_model'])[0]
        self.date_path = context['task_instance'].xcom_pull(key = 'date_path', task_ids = ['get_data'])[0]
        #MAPE
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred/y_true))) * 100
        metrics_df = pd.DataFrame()
        metrics_df['Model'] = ['SARIMAX', 'XGBoostRegressor']
        metrics_df['MSE'] = [
            metrics.mean_squared_error(self.sarimax_test, self.sarimax_pred),
            metrics.mean_squared_error(self.xgboost_y_test, self.xgboost_y_pred)]
        metrics_df['MAE'] = [
            metrics.mean_absolute_error(self.sarimax_test, self.sarimax_pred),
            metrics.mean_absolute_error(self.xgboost_y_test, self.xgboost_y_pred)]
        metrics_df['RMSE'] = [
            np.sqrt((metrics.mean_squared_error(self.sarimax_test, self.sarimax_pred))),
            np.sqrt((metrics.mean_squared_error(self.xgboost_y_test, self.xgboost_y_pred)))]
        metrics_df['MAPE'] = [
            mean_absolute_percentage_error(self.sarimax_test, self.sarimax_pred),
            mean_absolute_percentage_error(self.xgboost_y_test, self.xgboost_y_pred)]
        metrics_df['R2'] = [
            metrics.r2_score(self.sarimax_test, self.sarimax_pred),
            metrics.r2_score(self.xgboost_y_test, self.xgboost_y_pred)]
        fig = go.Figure(data = [go.Table(
            header = {
                'values': metrics_df.columns,
                'fill_color': 'paleturquoise'},
            cells = {
                'values': [metrics_df[column] for column in metrics_df.columns],
                'fill_color': 'lavender'})])
        fig.update_layout(title_text = 'METRICS')
        os.chdir(self.date_path)
        fig.write_html('metrics.html')



############################ Airflow DAG ##########################################
bp = BitcoinPredictions()

_requirements = BashOperator(
    task_id = 'requirements',
    bash_command = 'cd ~/airflow/dags && pip3 install -r dataeng2_requirements.txt',
    dag = dag)

_create_folder = PythonOperator(
    task_id = 'create_folder',
    python_callable = bp.create_folder,
    dag = dag)

_get_data = PythonOperator(
    task_id = 'get_data',
    python_callable = bp.get_data,
    provide_context = True,
    dag = dag)

_sarimax_model = PythonOperator(
    task_id = 'sarimax_model',
    python_callable = bp.sarimax_model,
    provide_context = True,
    dag = dag)

_xgboost_model = PythonOperator(
    task_id = 'xgboost_model',
    python_callable = bp.xgboost_model,
    provide_context = True,
    dag = dag)

_sarimax_plot = PythonOperator(
    task_id = 'sarimax_plot',
    python_callable = bp.sarimax_plot,
    provide_context = True,
    dag = dag)

_xgboost_plot = PythonOperator(
    task_id = 'xgboost_plot',
    python_callable = bp.xgboost_plot,
    provide_context = True,
    dag = dag)

_metrics = PythonOperator(
    task_id = 'metrics',
    python_callable = bp.metrics,
    provide_context = True,
    dag = dag)

_requirements >> _create_folder
_create_folder >> _get_data
_get_data >> [_sarimax_model, _xgboost_model]
_sarimax_model >> _sarimax_plot
_xgboost_model >> _xgboost_plot
[_sarimax_plot, _xgboost_plot] >> _metrics

########################################################################################

        