git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make

datapath=/vpublic01/frog/zhengzhigao/feluca4big/example

./vpublic01/frog/zhengzhigao/feluca_exp/exp ${datapath}/roadnet/roadnet.vertices ${datapath}/roadnet/roadnet.edges 735292 3523247 539587 1761624 4