#/bin/bash
touch ./valData/label.txt
touch ./trainData/label.txt
age=10
sex=1
mal=0
for i in {1..100}
do
  cp test.jpg ./valData/$i.jpg
  echo "$i $age $sex $mal" >> ./valData/label.txt
  cp test.jpg ./trainData/$i.jpg
  echo "$i $age $sex $mal" >> ./trainData/label.txt
done
