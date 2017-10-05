#!/bin/bash

#======================================================================================

#upload file to aws clusters
if [ $# -lt 2 ]; then
	echo "usage: ./upload_install_split.sh /path/to/codedir /path/to/datadir" 
	exit 0
fi

hosts="
54.91.23.131
54.81.74.99
52.90.246.238
54.91.123.123
54.88.79.0
"

rank=0
for ip in $hosts; 
do
    printf "upload code....\n"
    scp -o "StrictHostKeyChecking no"  -i ~/.ssh/Firstkey.pem $1aws_code.tar ubuntu@$ip:~/
  if [ $rank -ne 0 ]; then
    printf "upload data....\n"
    scp -o "StrictHostKeyChecking no"  -i ~/.ssh/Firstkey.pem $2aws_data_$rank.tar ubuntu@$ip:~/
  fi
  (( rank++ ))
done

rank=0
for ip in $hosts; 
do
    #extract code	
    ssh -o "StrictHostKeyChecking no"  -i ~/.ssh/Firstkey.pem  ubuntu@$ip "tar xvf /home/ubuntu/aws_code.tar"
  if [ $rank -ne 0 ]; then
    #extract data
    ssh -o "StrictHostKeyChecking no"  -i ~/.ssh/Firstkey.pem  ubuntu@$ip "tar xvf /home/ubuntu/aws_data_$rank.tar"
  fi
    #install
    ssh -o "StrictHostKeyChecking no"  -i ~/.ssh/Firstkey.pem  ubuntu@$ip "cd /home/ubuntu/MPIPlatform && ./install.sh"

    (( rank++ ))
done






