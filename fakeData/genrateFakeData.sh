#/bin/bash
touch ./valData/label.txt
touch ./trainData/label.txt
age=10
benign=1
mal=0
for i in {1..100}
do
  cp test.jpg ./valData/$i.jpg
  echo "$i.jpg 0 1" >> ./valData/label.txt
  cp test.jpg ./trainData/$i.jpg
  echo "$i.jpg 0 1" >> ./trainData/label.txt
done
for i in {1..100}
do
  cp test.jpg ./valData/$i.jpg
  echo "$i.jpg 1 0" >> ./valData/label.txt
  cp test.jpg ./trainData/$i.jpg
  echo "$i.jpg 1 0" >> ./trainData/label.txt
done
