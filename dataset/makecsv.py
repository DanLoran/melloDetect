import os
target = open("label.csv","a")
for file in os.listdir('Data/Descriptions'):
    f = open('Data/Descriptions/'+file)
    filetext = f.readlines()
    age = filetext[23].strip().split(":")[1]
    age = age.replace(' ','')
    age = age.replace(',','')
    sexstr = filetext[29].strip().split(":")[1]
    if("female" in sexstr):
        sex = 0
    elif("male" in sexstr):
        sex = 1
    else:
        sex = -1
    malstr = filetext[25].strip().split(":")[1]
    if("benign" in malstr):
        mal = 0
    elif("malignant" in malstr):
        mal = 1
    else:
        mal = -1
    target.write(file+ "," + str(age) + "," + str(sex) + "," + str(mal) + '\n')
    f.close()
    print("finished file " + file + '\n')
target.close()
