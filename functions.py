import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact

def symbol_extraction(dataset):

    symbols = [s for s in dataset['Symbol']]

    return symbols
    
def search_csv(etf_folders, root_dir):

    etfs_dataset_list = []
    etfs_category_names_list = []

    for folder in etf_folders:
        #print(folder)
        folder_dir = os.path.join(root_dir, folder)

        if os.path.isdir(folder_dir):

            for dirpath, dirname, dirfiles in os.walk(folder_dir):
                #print(dirfiles)

                for csv_name in dirfiles:
                    #print(csv_name)

                    if csv_name.lower().endswith('.csv'):
                        #print(csv_name)

                        dataset_name = f'etf_{folder}_{csv_name.split("_")[0]}'
                        etf_dataset = pd.read_csv(f'{folder_dir}\\{csv_name}')
                        etfs_dataset_list.append(etf_dataset)
                        etfs_category_names_list.append(dataset_name)
                    else:
                        pass
        else:
            return 'Could not find this path'
        
    return etfs_dataset_list,etfs_category_names_list


def calculate_rsi(data, tickers, list_window):

    df_temp = []

    for ticker in tickers:

        df_ticker = data[data['Ticker'] == ticker]
        

        for num in list_window:

            delta = df_ticker.Close.diff(num)

            gain = (delta.where(delta > 0, 0)).rolling(window=num).mean()
            
            loss = (-delta.where(delta < 0, 0)).rolling(window=num).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            df_ticker[f'RSI_{num}'] = rsi

            #print(ticker,df_ticker)
        df_temp.append(df_ticker)

    #print('Este es el df temporal:',df_temp)
    return df_temp


def calculate_ewma(data,tickers,list_span):
        
        df_temp = []

        for ticker in tickers:
                df_ticker = data[data.Ticker == ticker] 
                #print(df_ticker)
                for num in list_span:
                        #print(df_ticker['Close'].ewm(span=7, adjust=False).mean())
                        df_ticker[f'ewma_{num}'] = df_ticker['Close'].ewm(span=num, adjust=False).mean()

                #print('ESTE ES DF_TICKER',df_ticker)

                df_temp.append(df_ticker)

        return pd.concat(df_temp)
    
    
def calculate_macd(data):
    
    columns_to_drop = data.columns

    tickers = ['FGDL','SSO']
    emas_12_26 = calculate_ewma(data,tickers,[12,26]) #devuelve una lista con las columnas
    emas_12_26 = pd.concat(emas_12_26)
    emas_12_26 = emas_12_26.drop(columns_to_drop,axis=1)
    print('mis emas:',emas_12_26['ewma_12'])
    
    macd = emas_12_26['ewma_12'] - emas_12_26['ewma_26']
    print('MACD',macd)

    signal = macd.ewm(span=9, adjust=False).mean()
    print('SIGNAL', signal)
    data['MACD'] = signal

    return data

def data_train_test(series):


    pct_train = int(len(series) * 0.6)
    pct_val = int(len(series) * 0.2)
    

    train = series[:pct_train]
    val = series[pct_train:pct_train + pct_val]
    test = series[pct_train + pct_val:]
    
    return train, val, test


def data_train_test_regression(X, y):
    
    pct_train = int(len(X) * 0.6)
    pct_val = int(len(X) * 0.2)
    
    
    X_train, y_train = X[:pct_train], y[:pct_train]
    X_val, y_val = X[pct_train:pct_train + pct_val], y[pct_train:pct_train + pct_val]
    X_test, y_test = X[pct_train + pct_val:], y[pct_train + pct_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_lags(df, lags):

    for i in range(1,lags+1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    return df

def plots_predictions(data, 
                      n,
                      list_dates, 
                      y_train,
                      y_val, 
                      y_test,
                      metric1,
                      metric2):

    fig = make_subplots(rows=15, cols=2, subplot_titles=[f"Plot {i+1}" for i in range(n)])

 
    for i in range(n):
        row = i // 5 + 1
        col = i % 5 + 1

        fig.add_trace(
            go.Scatter(
                x=list_dates[0], y=y_train,
                mode='markers',
                name=f'Train {i+1}',
                marker=dict(color='blue')
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=list_dates[2], y=y_test,
                mode='markers',
                name=f'Test {i+1},\n MSE: {metric1}, \n R2: {metric2}',
                marker=dict(color='red')
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=list_dates[1], y=y_val,
                mode='markers',
                name=f'Validation {i+1}',
                marker=dict(color='green')
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=data['Date'], y=y_pred,
                mode='lines+markers',
                name=f'Test Predictions {i+1}',
                marker=dict(color='orange'),
                line=dict(dash='dash')
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=2000, width=1500, 
        title_text=f'Train, Test, Validation and Test Predictions for {n} Datasets',
        showlegend=False
    )

    return fig.show()

def clean_data(data):

    #Cleaning
    data.dropna(inplace=True)

    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data['Date'] = data['Date'].dt.date
    
    data_lags = create_lags(data, 7)
    data_lags.dropna(inplace=True)

    return data_lags

def corr_matrix(data):

    #Correlation matrix
    plt.figure(figsize=(15,10))
    corrmat = data.drop(['Ticker','Date'], axis=1).corr()
    sns.heatmap(corrmat, square=True)

    k = len(data.columns)

    cols = corrmat.nlargest(k, 'Close_target')['Close_target'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
    yticklabels=cols.values, xticklabels=cols.values)

    return plt.show()