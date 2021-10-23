#coding = utf-8
import os


def caulate(col_Line,team_index,contents):
    sum = 0
    num = 0
    for sample in contents[team_index:team_index + 23]:
        try:
            if sample.strip().split(',')[col_Line] == '0' or sample.strip().split(',')[col_Line] == 'NA':
                pass
        except IndexError:
            print(sample,col_Line)
        else:
            try:
                sum += float(sample.strip().split(',')[col_Line])
                num += 1
            except ValueError:
                pass
    if num == 0:
        ave =0
    else:
        ave = sum / num
    return ave

def parase_prediction_csv(file_path):

    with open(file_path,'r') as fp:
        contents = fp.readlines()
    team_data = []
    flag = '0'
    for index in range(1,25417):
        date_list = contents[index].strip().split(',')[1]
        place = contents[index].strip().split(',')[2]
        date = date_list.strip().split()[0]
        if flag != date:
            team_index = index
            team_data.append("{},{},".format(date,place))
            for col_line in range(3,23):
                ave = caulate(col_line,team_index,contents)
                team_data.append("{},".format(ave))
            team_data.append('\n')
        flag = date
    os.chdir('/Users/huhao/Documents/cv_project/建模/question4/')
    with open('A3_prediction_day.csv','w') as f:
        f.writelines(team_data)


def parase_true_csv(file_path):
    with open(file_path,'r') as fp:
        contents = fp.readlines()
    team_data = []
    flag = '0'
    for index in range(1, 19659):
        try:
            date = contents[index].strip().split()[0]
            place = contents[index].strip().split(',')[1]
        except IndexError:
            pass
        if flag != date:
            team_index = index
            team_data.append("{},{},".format(date, place))
            for col_line in range(2, 12):
                ave = caulate(col_line, team_index, contents)
                team_data.append("{},".format(ave))
            team_data.append('\n')
        flag = date
    os.chdir('/Users/huhao/Documents/cv_project/建模/question4/')

    with open('A3_true_day.csv', 'w') as f:
        f.writelines(team_data)



if __name__ == "__main__":
    file_path = '/Users/huhao/Documents/cv_project/建模/question4/A3_predict_hour.csv'
    parase_prediction_csv(file_path)
    #parase_true_csv(file_path)