nohup python -u train.py > ./log/train.log 2>&1 &
tail -f ./log/train.log