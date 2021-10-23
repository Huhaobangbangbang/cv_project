#coding = utf-8
import os


def caulate(col_Line,team_index,contents):
    sum = 0
    num = 0
    for sample in contents[team_index:team_index + 23]:
        if sample.strip().split(',')[col_Line] == 'NULL' or sample.strip().split(',')[col_Line] == 'NA':
            pass
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
    for index in range(1,25345):
        date_list = contents[index].strip().split()[0]
        place = contents[index].strip().split(',')[2]
        date = date_list.split(',')[1]
        if flag != date:
            team_index = index
            team_data.append("{},{},".format(date,place))
            for col_line in range(3,23):
                ave = caulate(col_line,team_index,contents)
                team_data.append("{},".format(ave))
            team_data.append('\n')
        flag = date
    os.chdir('/Users/huhao/Documents/cv_project/建模/question3/')
    with open('C_prediction_day.csv','w') as f:
        f.writelines(team_data)
def parase_true_csv(file_path):
    with open(file_path,'r') as fp:
        contents = fp.readlines()
    team_data = []
    flag = '0'
    for index in range(1, 25345):
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
    os.chdir('/Users/huhao/Documents/cv_project/建模/question3/')
    with open('B_true_day.csv', 'w') as f:
        f.writelines(team_data)



if __name__ == "__main__":
    file_path = '/Users/huhao/Documents/cv_project/建模/question3/B_true_hour.csv'
    #parase_prediction_csv(file_path)
    parase_true_csv(file_path)