# 比较预测的船舶通过位置和标签位置的距离，转换到时间和空间位置
import os
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import datetime


def noise_sample_mes(filename):
    pattern = r"(\d{8}-\d{6})_D(\d+\.\d+)_DT([-+]?\d+(?:\.\d+)?)_DD([-+]?\d+(?:\.\d+)?)_DS(\d+)"    
    match = re.match(pattern, filename) 
    if match:
        time = match.group(1)
        pos = match.group(2)  
        time_dist = match.group(3)
        space_dist = match.group(4)
        samples = match.group(5)
        return {'Time':[time],'Position':[pos],'DistInTime':[time_dist],'DistInSpace':[space_dist],'SamplesPerSec':[samples]}
    else:
        return None

def ship_sample_mes(filename):
    '''
    返回从船舶样本中分析的样本设置
    '''
    "20231116-160439_D12.09_DT1_DD0.5_DS20_Dt0.0_Ds0.0_S2.93_A91.71.png"
    #filename = "20210930-231841_D12.36_DT1_DD1_DS10_Dt0.0_Ds0.0_S6.05_A86.71.png"
    pattern = r"(\d{8}-\d{6})_D(\d+\.\d+)_DT([-+]?\d+(?:\.\d+)?)_DD([-+]?\d+(?:\.\d+)?)_DS(\d+)_Dt([-+]?\d+(?:\.\d+)?)_Ds(\d+\.\d+)_S(\d+\.\d+)_A(\d+\.\d+)"    
    match = re.match(pattern, filename) 
    if match:
        time = match.group(1)
        pos = match.group(2)  
        time_dist = match.group(3)
        space_dist = match.group(4)
        samples = match.group(5)
        time_bias = match.group(6)
        space_bias = match.group(7)
        speed = match.group(8)
        angle = match.group(9)
        return {'Time':[time],'Position':[pos],'DistInTime':[time_dist],'DistInSpace':[space_dist],'SamplesPerSec':[samples],'TimeBias':[time_bias],'SpaceBias':[space_bias],'ShipSpeed':[speed],'CrossAngle':[angle]}
    else:
        return None

def  get_centerTime(filename):
    if len(filename)>55:  #利用filename长度判断是否为船舶样本
        mes = ship_sample_mes(filename)
        CrossTime = datetime.datetime.strptime(mes['Time'][0],'%Y%m%d-%H%M%S')
        Time_bias = float(mes['TimeBias'][0])
        CenterTime = CrossTime+datetime.timedelta(minutes=Time_bias)
    else:
        mes = noise_sample_mes(filename)
        CenterTime = datetime.datetime.strptime(mes['Time'][0],'%Y%m%d-%H%M%S')
    return CenterTime

def cal_center_pos(cross_position,Channels):
    dist = [abs(cross_position-c) for c in Channels]
    slice_position = Channels[dist.index(min(dist))]
    return slice_position

def get_center_pos(filename,Channels):
    if len(filename)>55: 
        mes = ship_sample_mes(filename)
        crossPos = float(mes['Position'][0])
        SpaceBias = float(mes['SpaceBias'][0])
        CenterPos = cal_center_pos(crossPos+SpaceBias,Channels)
    else:
        mes = noise_sample_mes(filename)
        CenterPos = cal_center_pos(float(mes['Position'][0]),Channels)
    return CenterPos


def get_crosstime_pred(time_pred,CenterTime,deltaT):
    '''
    计算pred中label所对应的船舶通过时间和位置
    '''
    Total_min = 2*deltaT+1
    Time_bias = datetime.timedelta(minutes=Total_min*(time_pred-0.5))
    return CenterTime+Time_bias

def get_crosspos_pred(pos_pred,CenterPos,deltaD):
    cross_bias = 0.4  #0.4是空间误差补正，因为样本文件的建立是进行空间补正后，因此filename中关于Position的数值比AIS_ship(空间补正后)相差了0.4，这个应该后面的数据集重构时候一并修复
    Total_dist = deltaD*2
    Pos_bias = Total_dist*(pos_pred-0.5)
    return CenterPos+Pos_bias+cross_bias



detect_path = '/home/huangwj/DAS/Yolo/yolov5/runs/detect/exp/'
label_txt = os.listdir(detect_path+'labels')
print(len(label_txt))
pred_df = pd.DataFrame()
for fn in tqdm(label_txt):
    txt_path = os.path.join(detect_path+'labels',fn)
    pred_dict = {}
    with open(txt_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            numbers = [float(num) for num in line.split()]
            pred_dict['filename'] = [fn.replace('.txt','')]
            pred_dict['type_pred'] = [numbers[0]]
            pred_dict['x_pred'] = [numbers[1]]
            pred_dict['y_pred'] = [numbers[2]]
            pred_dict['w_pred'] = [numbers[3]]
            pred_dict['h_pred'] = [numbers[4]]
            tmp = pd.DataFrame(pred_dict)
            pred_df = pd.concat([pred_df,tmp],axis=0)
pred_df.reset_index(inplace=True)

#船舶数据集样本的设置
delta_T = 0  # 单位：分钟    
delta_D = 0.5    # 单位：km
MAX_DISTANCE = 12.7    # 岸上的最大距离(km),DAS数据最大通道位置
MIN_DISTANCE = 2.7    # 岸上的最小距离(km),DAS数据最小通道位置
Fiber_1 = 13.07 #Km 桂山岛起点，用于确定cross_pos
Fiber_0 = 4.5 #Km 三角岛起点，不同于MIN_Distance，Fiber_0表示是直线段的光纤，而非光纤数据的最小通道位置
Channels = [c for c in np.arange(4.5+delta_D, MAX_DISTANCE-delta_D,delta_D) ]  #数据切片中心通道
print(Channels)

#计算样本的中心时间

filename = pred_df.iloc[0]['filename']
pred_df['CenterTime'] = pred_df.apply(lambda row: get_centerTime(row['filename']),axis=1)
pred_df['CenterPos'] = pred_df.apply(lambda row: get_center_pos(row['filename'],Channels),axis=1)

pred_df['CrossTime_pred'] = pred_df.apply(lambda row: get_crosstime_pred(row['x_pred'],row['CenterTime'],delta_T),axis=1)
pred_df['CrossPos_pred'] = pred_df.apply(lambda row: get_crosspos_pred(row['y_pred'],row['CenterPos'],delta_D),axis=1)

print(pred_df)



#predict结果与ais记录进行校验------------------------------
AIS_ship  =  pd.read_csv('/home/huangwj/DAS/Ship_Detect_Analyze_System/send_ais_data/FiberBoatMessage_all_history.csv',index_col=0)
AIS_ship['CrossTime'] = pd.to_datetime(AIS_ship['CrossTime'],format='%Y-%m-%d %H:%M:%S')
AIS_ship.sort_values(by=['CrossTime'],ascending=True,inplace=True)
err_time = 5  #second
err_pos = 0.2 #km
print(AIS_ship.columns)

def compare_with_ais(AIS_ship,CrossTime_pred,CrossPos_pred,err_time,err_pos):
    st = CrossTime_pred+datetime.timedelta(minutes=-1)
    et = CrossTime_pred+datetime.timedelta(minutes=1)
    cmp_df = AIS_ship[(AIS_ship['CrossTime']>st)&(AIS_ship['CrossTime']<et)]
    if len(cmp_df)>0:
        cmp_num = 0
        POS = -1
        for i in range(0,len(cmp_df)):
            row = cmp_df.iloc[i]
            if abs((row['CrossTime']-CrossTime_pred).total_seconds())<=err_time:
                POS = row['cross_position']
                if abs(row['cross_position']-CrossPos_pred)<=err_pos:
                    cmp_num = 1
                    break
    else:
        cmp_num = 0
        POS = -1
    return cmp_num,POS

pred_df['cmp_ais'],pred_df['crossPos_ais'] = zip(*pred_df.apply(lambda row: compare_with_ais(AIS_ship,row['CrossTime_pred'],row['CrossPos_pred'],err_time,err_pos),axis=1))
print(pred_df['cmp_ais'].value_counts())
pred_df.to_csv('cmp_with_ais.csv')