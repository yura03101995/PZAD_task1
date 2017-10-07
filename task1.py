import numpy as np
import pandas as pd
import scipy.stats as spstats
from scipy.sparse import csr_matrix
import numpy.random as nprand

global_count_week = 0

def d_sol(data,offset_week=0, aprx_deg=4,count_weeks = 62):
    ss = csr_matrix((data['sum'].values, (data.id.values - 1, data.date.values - 1)))
    a = []
    for i in range(110000):
        sh = ss[i,:].toarray().ravel()
        h = sh[1:].reshape(-1, 7)
        g = (((h>0).cumsum(axis=1) == 1) * h).sum(axis=1)
        j = np.argmax(np.dot ( np.arange( count_weeks ), 
            csr_matrix((np.ones( count_weeks ), 
            (np.arange( count_weeks ), g)), 
            shape=( count_weeks, 17)).toarray()))
        a.append(j)
    df = pd.DataFrame({'id': np.arange(1, 110001), 'sum':a})
    max_day = data['date'].max()
    df_max_diff_id = data.set_index('id')['date'].diff().groupby('id').max().to_frame().reset_index(drop=True)
    df_diff_btw_max_last = data.groupby('id')['date'].max().to_frame().reset_index(drop=True)
    df_diff_btw_max_last['date'] = max_day - df_diff_btw_max_last['date']
    df_id_out = (df_diff_btw_max_last['date'] < aprx_deg ).to_frame()
    '''df_max_diff_id['date']'''
    df_id_out = df_id_out.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df['sum'] = df['sum']*df_id_out['date']
    return df

def get_prediction(df,offset_week=0,aprx_deg=4,count_weeks=22):
    median = df.groupby(['id'])['sum'].median(skipna=True)
    df_median = median.to_frame()
    df_median['sum'] = df_median['sum'].astype(int)
    max_day = df['date'].max()
    df_max_diff_id = df.set_index('id')['date'].diff().groupby('id').max().to_frame()
    df_diff_btw_max_last = df.groupby('id')['date'].max().to_frame()
    df_diff_btw_max_last['date'] = max_day - df_diff_btw_max_last['date']
    df_id_out = (df_diff_btw_max_last['date'] < 5*7).to_frame()
    '''df_max_diff_id['date']'''
    df_median['sum'] = df_median['sum'] * df_id_out['date']
    return df_median

def testing_prediction(fn_getter_prediction, df, offset_week, aprx_deg=6, count_weeks=30):
    max_day = df['date'].max()
    train_days = max_day - 7 * offset_week
    df_train = df[ df['date'] <= train_days ].copy()
    train_days_up = max_day - 7 * (offset_week - 1)
    df_test = df[ df['date'] <= train_days_up ].copy()
    df_test = df_test[ df_test['date'] > train_days ]
    df_test = df_test.groupby('id',as_index=False).head(1)
    df_test = df_test[ ['id','sum'] ]
    full = { k for k in range(1,110001) }
    part_ids = set(df_test['id'].unique())
    non_rm_ids = full.difference( part_ids )
    df_for_append = pd.DataFrame([(i,0) for i in non_rm_ids],columns=['id','sum']).set_index('id')
    df_test = df_test.set_index('id')
    df_test = df_test.append(df_for_append).sort_index()
    df_test['id'] = df_test.index
    df_test = df_test.reset_index(drop=True)
    df_predict = fn_getter_prediction( df_train, offset_week, aprx_deg, count_weeks )
    df_predict['id'] = df_predict.index
    full = { k for k in range(1,110001) }
    part_ids = set(df_predict['id'].unique())
    non_rm_ids = full.difference( part_ids )
    df_for_append = pd.DataFrame([(i,0) for i in non_rm_ids],columns=['id','sum']).set_index('id')
    df_predict = df_predict.set_index('id')
    df_predict = df_predict.append(df_for_append).sort_index()
    df_predict = df_predict.reset_index(drop=True)
    df_predict = df_predict[ df_predict.index < 110000 ]
    diff = {k:0 for k in range(0,11)}
    for i in range(0,11):
        part_test = df_test[ 'sum' ].value_counts( normalize=True ).get(i)
        part_pred = df_predict[ 'sum' ].value_counts( normalize=True ).get(i)
        if( part_test is None ):
            part_test = 0
        if( part_pred is None ):
            part_pred = 0
        diff[i] += abs(part_test - part_pred )
    #print(diff)
    #print(df_test[ 'sum' ].value_counts( normalize=True ).get(0))
    return ( df_test[ 'sum' ] == df_predict[ 'sum' ] ).value_counts(normalize=True).get(True)

def super_predict(df, offset_week=0, aprx_deg=6, count_weeks=30):
    #aprx_deg = 6
    #count_weeks = 30
    h_mul = 1.3
    df_without_nose = df[ df['date'] >= 355 - count_weeks * 7 ].copy()
    df_without_nose = df.copy()
    max_day = df_without_nose['date'].max()
    mass = { j: list() for j in range(0,17)}
    df1 = pd.DataFrame({'key':[1 for i in range(1,110001)],'id':[i for i in range(1,110001)]})
    df2 = pd.DataFrame({'key':[1 for i in range(0,11)],'sum':[i for i in range(0,11)]})
    df_with_p = pd.merge(df1,df2,on='key')[['id','sum']]
    df_with_p['count'] = 0
    for i in range(count_weeks,0,-1):
        train_days_up = max_day - 7 * (i - 1)
        train_days = max_day - 7 * i
        df_test = df_without_nose[ df_without_nose['date'] <= train_days_up ].copy()
        df_test = df_test[ df_test['date'] > train_days ]
        df_test = df_test.groupby('id',as_index=False).head(1)
        df_test = df_test[ ['id','sum'] ]
        full = { k for k in range(1,110001) }
        part_ids = set(df_test['id'].unique())
        non_rm_ids = full.difference( part_ids )
        df_for_append = pd.DataFrame([(i,0) for i in non_rm_ids],columns=['id','sum']).set_index('id')
        df_test = df_test.set_index('id')
        df_test = df_test.append(df_for_append).sort_index()
        df_test['id'] = df_test.index
        indexes = pd.merge(df_with_p.reset_index(), df_test, how='inner').set_index('index').index
        df_with_p.loc[indexes,'count'] += pow( (11 - i), 0.3)
        for j in range(0, 17):
            percent = df_test[ 'sum' ].value_counts(normalize=True).get(j)
            if( percent is None ):
                percent = 0
            mass[j].append(percent)
    mass_er = {k:mass[k] for k in [i for i in range(0,11)] if k in mass}
    parts = { k: 0 for k in range(0,len(mass_er)) }
    for k in range(0,len(mass_er)):
        z = np.polyfit([i for i in range(0,count_weeks)],mass_er[k],deg=aprx_deg)
        pol = np.poly1d(z)
        parts[k] = pol(count_weeks)
    #print(parts)
    idx = df_with_p.groupby('id')['count'].transform(max) == df_with_p['count']
    df_loc_max = df_with_p.loc[idx]
    idx = df_loc_max.groupby('id')['sum'].transform(min) == df_loc_max['sum']
    df_loc_max = df_loc_max.loc[idx]

    df_return = pd.DataFrame(columns=['id','sum','count'])
    for k in range(0,10):
        df_loc_max_by_sum = df_loc_max[ df_loc_max['sum'] == k ].sort_values('count')
        df_loc_max_by_sum = df_loc_max_by_sum.reset_index(drop=True)
        needed_count = int(parts[k] * 110000 * h_mul)
        treshold = len(df_loc_max_by_sum.index) - needed_count
        df_loc_max_by_sum = df_loc_max_by_sum[ df_loc_max_by_sum.index > treshold ]
        df_return = df_return.append(df_loc_max_by_sum)
    
    full = { k for k in range(1,110001) }
    part_ids = set(df_return['id'].unique())
    non_rm_ids = full.difference( part_ids )

    df_with_p_c = df_with_p.copy()
    df_with_p_c = df_with_p_c[ df_with_p_c['id'] == non_rm_ids]
    idx = df_with_p_c.groupby('id')['count'].transform(max) == df_with_p_c['count']
    df_with_p_c.loc[idx,'count'] = 0
    idx = df_with_p_c.groupby('id')['count'].transform(max) == df_with_p_c['count']
    df_loc_max_2 = df_with_p_c.loc[idx]
    idx = df_loc_max_2.groupby('id')['sum'].transform(min) == df_loc_max_2['sum']
    df_loc_max_2 = df_loc_max_2.loc[idx]
    for k in range(0,10):
        df_loc_max_by_sum = df_loc_max[ df_loc_max['sum'] == k ].sort_values('count')
        df_loc_max_by_sum = df_loc_max_by_sum.reset_index(drop=True)
        needed_count = int(parts[k] * 110000)
        treshold = len(df_loc_max_by_sum.index) - needed_count
        if treshold < 0:
            df_loc_max_2_by_sum = df_loc_max_2[ df_loc_max_2['sum'] == k ].sort_values('count')
            df_loc_max_2_by_sum = df_loc_max_2_by_sum.reset_index(drop=True)
            treshold += (len(df_loc_max_2_by_sum.index))
            df_loc_max_2_by_sum = df_loc_max_2_by_sum[ df_loc_max_2_by_sum.index > treshold ]
            df_return = df_return.append(df_loc_max_2_by_sum)
    '''
    df_return = pd.DataFrame(columns=['id','sum','count'])
    df_with_p_c = df_with_p.copy()
    for i in range(0,10):
        idx = df_with_p_c.groupby('count')
        needed_counts = int(parts[i] * 110000)
        getter_count = 0
        for k in range(10,-1,-1):
            #print(df_with_p_c)
            df_group = idx.get_group(k)
            idx_loc = df_group.groupby('id')['sum'].transform(np.median).astype(int) == df_group['sum']
            df_group = df_group.loc[idx_loc]
            df_group = df_group[ df_group['sum'] == i ]
            len_group = len(df_group.index)
            if(len_group+getter_count > needed_counts):
                df_group.reset_index(drop=True)
                df_group = df_group[ df_group.index < needed_counts - getter_count ]
                df_return = df_return.append(df_group)
                break
            df_return = df_return.append(df_group)
            getter_count += len_group
            #full_idx = set(df_with_p_c.index)
            #erase_idx = set(df_group.index)
            #non_rm_idx = full_idx.difference(erase_idx)
            #df_with_p_c = df_with_p_c.loc[ non_rm_idx ]
            
    df_return['sum'] = df_return['sum'].astype(int)
    idx_loc = df_return.groupby('id')['sum'].transform(np.median).astype(int) == df_return['sum']
    df_return = df_return.loc[idx_loc]
    '''
    full = { k for k in range(1,110001) }
    part_ids = set(df_return['id'].unique())
    non_rm_ids = full.difference( part_ids )
    #df_return = df_return.sort_values('id').reset_index(drop=True)[['id','sum']]
    #df_loc_max = df_loc_max.set_index('id')
    #df_for_append = df_loc_max.loc[non_rm_ids][['sum']]
    df_for_append = pd.DataFrame([(i,0) for i in non_rm_ids],columns=['id','sum']).set_index('id')
    df_return = df_return.set_index('id')
    df_return = df_return.append(df_for_append).sort_index()
    df_return['id'] = df_return.index
    return df_return.reset_index(drop=True)
    
def main():
    global global_count_week
    df = pd.read_csv("C:/MMP_MSU/PZAD/PZAD_task1/train2.csv")
    #f = open("ans.txt",'w')
    df_after = df[ df[ 'date' ] > 438 - 20 * 7 ]
    df_before = df[ df[ 'date' ] < 438 - 32 * 7 ]
    #print(df_after[df_after['id']==1])
    df_after['date'] = df_after['date'] - 12 * 7 + 1
    #print(df_before[df_before['id']==1])
    df_before = df_before.append( df_after )
    df = df_before
    #print(df_before[df_before['id']==1])
    global_count_week = 10
    #print(testing_prediction( super_predict, df, 1, 4, 22 ))
    #df_ret = super_predict(df,0,4,22)
    #print(df_ret)
    #df_ret.to_csv("C:/MMP_MSU/PZAD/PZAD_task1/sup_res.csv", index=False)
    max_pred = 0
    max_size = 0
    size = 50
    preds = []
    degs = []
    for deg in range(18,11,-1):
        df_without_nose = df[ df['date'] >= 355 - size * 7 ].copy()
        df_without_nose['date'] = df_without_nose[ 'date' ] - ( 355 - size * 7) + 1
        full = { k for k in range(1,110001) }
        part_ids = set(df_without_nose['id'].unique())
        non_rm_ids = full.difference( part_ids )
        if non_rm_ids:
            df_for_append = pd.DataFrame([(i,10,0) for i in non_rm_ids],columns=['id','date','sum'])
            df_without_nose = df_without_nose.append(df_for_append).sort_values(['id','date'])
        df_without_nose = df_without_nose.reset_index(drop=True)
        #df_ret = d_sol(df_without_nose,0,deg,size)
        #df_ret.to_csv("C:/MMP_MSU/PZAD/PZAD_task1/sup_res.csv", index=False)
        pred = testing_prediction( d_sol, df_without_nose, 1, deg, size - 1 )
        print(pred,deg)
        preds.append(pred)
        degs.append(deg)
        #f.write(str(pred) + " " + str(size)+'\n')
        #if(pred > max_pred):
        #    max_pred = pred
        #    max_size = size
    print(preds)
    print(degs)
    #f.write(str(max_pred) + " " + str(max_size))
    #f.close()
    #print( testing_prediction( get_prediction, df_without_nose, 1 ) )
    #df_without_nose = df_without_nose.reset_index()
    '''
    max_pred = 0
    max_aprx = 0
    max_size = 0
    for j in range(25,11,-1):
        for i in range(2,6):
            print('aprx_deg=',i,'size=',j)
            loc_pred = testing_prediction( super_predict, df, 1, i, j )
            print('loc_pred=', loc_pred, "\n\n")
            if(loc_pred>max_pred):
                max_pred = loc_pred
                max_aprx = i
                max_size = j
    print(max_pred,max_aprx,max_size)
    '''
def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

main()