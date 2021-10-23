#coding = utf-8
import os
#coding = utf-8

def get_average(csv_path,index):
    with open(csv_path, 'r') as f:
        contents = f.readlines()
    num = 0
    sum = 0
    for sample in contents:
        air_data = sample.strip().split(',')[index]
        if air_data =='NA':
            pass
        else:
            num+=1
            sum += float(air_data)
    ave = sum/num
    return ave


def get_new_data(csv_path,so2_ave,no2_ave,pm10_ave,pm25_ave,o3_ave,co_ave):
    dataset_list = []
    with open(csv_path,'r') as f:
        contents = f.readlines()
    dataset_list.append("监测日期,地点,SO2监测浓度(μg/m³),NO2监测浓度(μg/m³),PM10监测浓度(μg/m³),PM2.5监测浓度(μg/m³),O3最大八小时滑动平均监测浓度(μg/m³),CO监测浓度(mg/m³)\n")
    for sample in contents:
        date = sample.strip().split(',')[0]
        place = sample.strip().split(',')[1]
        so2 = sample.strip().split(',')[2]
        if so2 =='NA':
            so2 = so2_ave
        no2 = sample.strip().split(',')[3]
        if no2 == 'NA':
            no2 = no2_ave
        pm10 = sample.strip().split(',')[4]
        if pm10 == 'NA':
            pm10 = pm10_ave
        pm25 = sample.strip().split(',')[5]
        if pm25 =='NA':
            pm25 = pm25_ave
        o3 = sample.strip().split(',')[6]
        if o3 == 'NA':
            o3 = o3_ave
        co = sample.strip().split(',')[7]
        if co=='NA':
            co = co_ave
        dataset_list.append("{},{},{},{},{},{},{},{}\n".format(date,place,so2,no2,pm10,pm25,o3,co))
    os.chdir('/Users/huhao/Downloads/model_test/')
    with open('que3_dataset_end.csv','w') as fp:
        fp.writelines(dataset_list)


def remove_bad_data():


if __name__=="__main__":
    csv_path = '/Users/huhao/Downloads/model_test/que3_dataset.csv'
    so2_ave = get_average(csv_path,2)
    no2_ave = get_average(csv_path,3)
    pm10_ave = get_average(csv_path,4)
    pm25_ave = get_average(csv_path,5)
    o3_ave = get_average(csv_path,6)
    co_ave = get_average(csv_path,7)
    get_new_data(csv_path, so2_ave, no2_ave, pm10_ave, pm25_ave, o3_ave, co_ave)

