#处理ship_detect_experiment.py所产生的结果文件"/sda1/huangwj/DAS/DataSet/local_samples/pred.csv"，将多条相似记录融合成单一的记录，统计船舶通行事件的数量

import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
pred_csv = "/sda1/huangwj/DAS/DataSet/local_samples/pred.csv"
Pred = pd.read_csv(pred_csv,index_col=0)

Pred['CrossTime_pred'] = pd.to_datetime(Pred['CrossTime_pred'],format = '%Y-%m-%d %H:%M:%S')
Pred['CenterTime'] = pd.to_datetime(Pred['CenterTime'],format = '%Y-%m-%d %H:%M:%S')
Pred['CrossTime'] = pd.to_datetime(Pred['CrossTime'],format = '%Y-%m-%d %H:%M:%S')

Pred.sort_values(by = 'CrossTime',ascending= True,inplace=True)
Pred.reset_index(inplace=True)

Pred['Used']=0

records_detected_num = 3  # 符合约定时间和空间误差的记录数量，主要用来筛选误检的记录
Err_Sec = 30
Err_Km = 0.15


ship_ls = []
for i in range(0,len(Pred)):
    record = Pred.iloc[i]
    if record['Used'] == 0:
        Ct= record['CrossTime']
        Ct_pred = record['CrossTime_pred']
        Cp_pred = record['CrossPos_pred']
        tmp = Pred[(Pred['CrossTime']<=(Ct+timedelta(minutes=2)))&(Pred['Used']==0)]
        tmp = tmp[(((tmp['CrossTime_pred']-Ct_pred).dt.total_seconds())<=Err_Sec)&(abs(tmp['CrossPos_pred']-Cp_pred)<Err_Km)]
        idx = tmp.index
        Pred.loc[Pred.index.isin(idx),'Used']=1
        #print(Pred[Pred.index.isin(idx)])
        prob_max_df = Pred[Pred.index.isin(idx)]
        ship_record = prob_max_df.sort_values(by = 'prob',ascending = False).iloc[0]
        if len(tmp)>=records_detected_num:
            ship_ls.append(ship_record)
        
        
ship_df = pd.DataFrame(ship_ls)
ship_df.reset_index(inplace=True)
ship_df[['filename', 'type_pred', 'x_pred', 'y_pred',
       'w_pred', 'h_pred', 'CrossTime', 'CrossPos', 'CenterTime', 'CenterPos',
       'CrossTime_pred', 'CrossPos_pred', 'class_n', 'prob', 'Used']].to_csv('ship_yolo.csv')

#后期需要更改为船舶检测的模型，而不是现在的船速检测模型

