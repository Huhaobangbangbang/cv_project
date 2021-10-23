#coding = utf-8
import os

def get_AQI_primary(air,air_list):
    # 计算出特定空气的AQI
    IAQI_list = [0,50,100,150,200,300,400,500]
    flag = 0 # 表示比第几个数大
    for index in range(len(air_list)):
        if float(air) < air_list[index]:
            flag=index
            break
    print(flag)
    bphi = air_list[flag]
    lo = air_list[flag-1]
    iaqhi = IAQI_list[flag]
    iaqlo = IAQI_list[flag-1]
    IAQ_air = (iaqhi - iaqlo)*(float(air) - lo)/(bphi - lo) + iaqlo

    return IAQ_air


def get_table1_data(A_path):
    with open(A_path, 'r') as fp:
        contents = fp.readlines()
    co2_list = [0, 2, 4, 14, 24, 36, 48, 60]
    so2_list = [0, 50, 150, 475, 800, 1600, 2100, 2600, 2620]
    no2_list = [0,40,80,180,280,565,750,940]
    o3_list = [0,100,160,215,265,800]
    pm10_list = [0,50,150,250,350,420,500,600]
    pm25_list = [0,35,75,115,150,250,350,500]
    end_result_list = []
    for sample in contents:
        date = sample.strip().split(',')[0]
        so2 = sample.strip().split(',')[1]
        no2 = sample.strip().split(',')[2]
        pm10 = sample.strip().split(',')[3]
        pm25 = sample.strip().split(',')[4]
        o3 = sample.strip().split(',')[5]
        co = sample.strip().split(',')[6]
        IAQ_co = get_AQI_primary(co,co2_list)
        IAQ_so2 = get_AQI_primary(so2,so2_list)
        IAQ_no2 = get_AQI_primary(no2,no2_list)
        IAQ_pm10 =  get_AQI_primary(pm10,pm10_list)
        IAQ_pm25 = get_AQI_primary(pm25,pm25_list)
        IAQ_o3 = get_AQI_primary(o3,o3_list)
        iaq_max = max(IAQ_co,IAQ_so2,IAQ_no2,IAQ_pm10,IAQ_pm25,IAQ_o3)
        if iaq_max>50:
            print(IAQ_so2,IAQ_no2,IAQ_pm10,IAQ_pm25,IAQ_o3,IAQ_co,iaq_max)
        end_result_list.append("{},{},{},{},{},{},{}\n".format(date,IAQ_so2,IAQ_no2,IAQ_pm10,IAQ_pm25,IAQ_o3,IAQ_co))
        os.chdir('/Users/huhao/Downloads/model_test/')
        with open('result1.csv','w') as f:
            f.writelines(end_result_list)


if __name__=="__main__":
    A_path = '/Users/huhao/Downloads/model_test/end_data1.csv'
    get_table1_data(A_path)