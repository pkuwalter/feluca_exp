git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make

datapath=/vpublic01/frog/zhengzhigao/feluca4big/example

./exp ${datapath}/dataset-4/output.vertices ${datapath}/dataset-4/output.edges 6 8 3 2 4