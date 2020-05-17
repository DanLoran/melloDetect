import os
import re

def MakeCSV(labelList, imageList):
    target = open("label.csv","a")

    processed = 0

    for file in labelList:

        # check if the label can be found in the image list
        imagename = file + '.jpeg'
        if imagename not in imageList:
            continue

        f = open('Data/Descriptions/'+file)
        filetext = f.readlines()

        targetstr = 'age_approx'
        res = [x for x in filetext if re.search(targetstr, x)]

        if res == []:
            continue

        age = res[0].strip().split(":")[1]
        age = age.replace(' ','')
        age = age.replace(',','')
        if(age == 'null'): age = -1

        targetstr = 'sex'
        res = [x for x in filetext if re.search(targetstr, x)]
        if("female" in res[0]):
            sex = 0
        elif("male" in res[0]):
            sex = 1
        else:
            sex = -1

        targetstr = 'benign_malignant'
        res = [x for x in filetext if re.search(targetstr, x)]
        if res == []:
            continue

        if('"benign"' in res[0]):
            mal = 0
        elif('"malignant"' in res[0]):
            mal = 1
        else:
            continue

        target.write(file+ "," + str(age) + "," + str(sex) + "," + str(mal) + '\n')
        f.close()

        processed += 1

    target.close()
    return processed

if __name__ == '__main__':
    # to execute the file standalone
    imageList = os.listdir('Data/Images')
    labelList = sorted(os.listdir('Data/Descriptions'))
    MakeCSV(labelList, imageList)
