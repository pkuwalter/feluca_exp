git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make

datapath=/vpublic01/frog/zhengzhigao/feluca4big/example

./exp ${datapath}/amazon/amazon.vertices ${datapath}/amazon/amazon.edges 735292 3523247 539587 1761624 4