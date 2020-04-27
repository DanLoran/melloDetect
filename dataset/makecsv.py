import os
import re
target = open("label.csv","a")
for file in sorted(os.listdir('Data/Descriptions')):
    f = open('Data/Descriptions/'+file)
    filetext = f.readlines()

    targetstr = 'age_approx'
    res = [x for x in filetext if re.search(targetstr, x)]
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
    if('"benign"' in res[0]):
        mal = 0
    elif('"malignant"' in res[0]):
        mal = 1
    else:
        mal = -1
    target.write(file+ "," + str(age) + "," + str(sex) + "," + str(mal) + '\n')
    f.close()
    print("finished file " + file + '\n')
target.close()
