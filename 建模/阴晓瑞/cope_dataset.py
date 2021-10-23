import os

def get_the_date_in_fujian4(fu4_path):
    with open(fu4_path,'r') as f:
        contents = f.readlines()
    dataset = []
    # range(0,10)表示有十组数据
    for team_index in range(0,10):

        for index in range(1,5):
            sample_data = contents[team_index * 6+index].strip().split(':')
            data6 = sample_data[5]
            dataset.append('{},'.format(data6))
        dataset.append('0\n')
    os.chdir('/Users/huhao/Documents/cv_project/建模/阴晓瑞/')
    with open('fujian4_data.txt','w') as f:
        f.writelines(dataset)



if __name__ == "__main__":
    fu4_path = '/Users/huhao/Documents/cv_project/建模/阴晓瑞/fujian4.txt'
    get_the_date_in_fujian4(fu4_path)