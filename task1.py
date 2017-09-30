import numpy as np
import pandas as pd

def get_prediction(df):
    median = df.groupby(['id'])['sum'].median(skipna=True)
    df_median = median.to_frame()
    df_median['sum'] = df_median['sum'].astype(int)
    max_day = df['date'].max()
    df_max_diff_id = df.set_index('id')['date'].diff().groupby('id').max().to_frame()
    df_diff_btw_max_last = df.groupby('id')['date'].max().to_frame()
    df_diff_btw_max_last['date'] = max_day - df_diff_btw_max_last['date']
    df_id_out = (df_diff_btw_max_last['date'] < df_max_diff_id['date']).to_frame()
    df_median['sum'] = df_median['sum'] * df_id_out['date']
    return df_median

def testing_prediction(fn_getter_prediction, df, offset_week):
    max_day = df['date'].max()
    train_days = max_day - 7 * offset_week
    df_train = df[ df['date'] <= train_days ].copy()
    df_test = df[ df['date'] > train_days ].copy()
    df_test = df.groupby('id', as_index=False).tail(1).copy()
    df_test.loc[df_test['date'] <= train_days, 'sum'] = 0
    df_test = df_test[ ['id', 'sum'] ]
    df_test = df_test.set_index('id')
    df_predict = fn_getter_prediction( df_train )
    #print( df_test[ 'sum' ].value_counts( normalize=True ).get(0) )
    #print( df_predict[ 'sum' ].value_counts( normalize=True ).get(0) )
    return ( df_test[ 'sum' ] == df_predict[ 'sum' ] ).value_counts(normalize=True).get(True)

df = pd.read_csv("C:/MMP_MSU/PZAD/PZAD_task1/train2.csv")
print( testing_prediction( get_prediction, df, 1 ) )
