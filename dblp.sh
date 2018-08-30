git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make

datapath=/vpublic01/frog/zhengzhigao/feluca4big/example

./exp ${datapath}/dblp/dblp.vertices ${datapath}/dblp/dblp.edges 933258 3353618 587218 1676809 4