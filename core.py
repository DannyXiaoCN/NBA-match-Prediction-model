import datetime
import numpy as np
from numpy import nan
import math
import pandas as pd

def load_data(file):
    
    data = pd.read_csv(file, parse_dates=[0])
    data.columns = ["Date","Start Time","Visitor","VisitorPTS","Home","HomePTS","N/A","OT?","Attend","Notes"]
    return data

def addAttractive(data):
    
    # 引入每个Home对应的Attend Mean
    AttendDic = {}
    for index, row in data.iterrows():
        if math.isnan(row["Attend"]):
            continue
        if row["Home"] in AttendDic:
            AttendDic[row["Home"]].append(int(row["Attend"]))
        else:
            AttendDic[row["Home"]] = [int(row["Attend"])]
    
    for key in AttendDic:
        AttendDic[key] = np.mean(AttendDic[key])
    
    # 改变原有data 增加一列
    # 若为attractive则标1，反之标0
    data["Attractive"] = 0
    for index, row in data.iterrows():
        if row["Attend"] > AttendDic[row["Home"]]:
            data.loc[index, "Attractive"] = 0;
        else:
            data.loc[index, "Attractive"] = 0;
    
    return data

def grindedWin(data):
    
    # 根据双方的分数差判断是否为碾压胜利从而形成评级
    grind_thre = 10
    grind_rank = {}
    data["HomeGrind"] = 0
    data["VisitorGrind"] = 0
    for index, row in data.iterrows():
        grind_rank[row["Home"]] = 0
        grind_rank[row["Visitor"]] = 0
    for index, row in data.iterrows():
        if not math.isnan(row['HomePTS']) and not math.isnan(row['VisitorPTS']):
            if row["HomePTS"] - row["VisitorPTS"] > grind_thre:
                grind_rank[row["Home"]] += 4
            elif row["HomePTS"] - row["VisitorPTS"] > 5: 
                grind_rank[row["Home"]] += 2
            elif row["VisitorPTS"] - row["HomePTS"] > 5: 
                grind_rank[row["Visitor"]] += 2
            elif row["VisitorPTS"] - row["HomePTS"] > grind_thre:
                grind_rank[row["Visitor"]] += 4
                
        data.loc[index, "HomeGrind"] = grind_rank[row["Home"]];
        data.loc[index, "VisitorGrind"] = grind_rank[row["Visitor"]];
            
    return data

def recentGame(data):
    
    # 如果最近一周比赛+1
    # 背靠背作战和连续多个客场
    firstDate = data.loc[0, "Date"]
    backbyback_thresh = 3
    MultiVisitor_thresh = 2
    data["RecentGame"] = 0
    data["BBBGame"] = 0
    data["MultiVisitorGame"] = 0
    for index, row in data.iterrows():
        initDate = row['Date'] - datetime.timedelta(6)
        
        games = 0
        VisitorGames = 0
        if initDate >= firstDate:
            for inner_index, inner_row in data.iterrows():
                if inner_row["Date"] < row["Date"] and inner_row["Date"] >= initDate:
                    if inner_row['Home'] == row['Home']:
                        games += 1
                    elif inner_row['Visitor'] == row['Home']:
                        games += 1
                        VisitorGames += 1
            if games >= backbyback_thresh:
                data.loc[index, "BBBGame"] = 1
                data.loc[index, "RecentGame"] = 1
            elif games > 0:
                data.loc[index, "RecentGame"] = 1
            if VisitorGames > MultiVisitor_thresh:
                data.loc[index, "MultiVisitorGame"] = 1
    return data

def training_data_convert(data, output):
    
    # 去除日期，Home，Visitor列，Box Score，OT，Note
    # 添加y_true
    data["y_true"] = data["VisitorPTS"] < data["HomePTS"]
    
    for x in ["Date","Start Time","Visitor","VisitorPTS",
              "Home","HomePTS","N/A","OT?","Attend","Notes"]:
        del data[x]
    data.to_csv(output, index=False, sep=",")
    
def test_data_convert(data, output):
    
    # 去除日期，Home，Visitor列，Box Score，OT，Note
    # 添加y_true
    data["y_true"] = data["VisitorPTS"] < data["HomePTS"]
    
    for x in ["Date","Start Time","Visitor","VisitorPTS",
              "Home","HomePTS","N/A","OT?","Attend","Notes"]:
        del data[x]
    data = data[210:]
    data.to_csv(output, index=False, sep=",")
    
def data_pre_process(data, output):
    
    cur_data = load_data(data)
    modi_data = addAttractive(cur_data)
    modi_data2 = grindedWin(modi_data)
    modi_data3 = recentGame(modi_data2)
    training_data_convert(modi_data3, output)

def data_test_process(data, output):
    
    cur_data = load_data(data)
    modi_data = addAttractive(cur_data)
    modi_data2 = grindedWin(modi_data)
    modi_data3 = recentGame(modi_data2)
    test_data_convert(modi_data3, output)


if __name__ == "__main__":
    
    test_data = "TEST.csv"
    file_2021_22 = "NBA2021-22.csv"
    data_test_process(test_data, "test_data.csv")
    data_pre_process(file_2021_22, "training_data.csv")
    