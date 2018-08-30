git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make
datapath=/vpublic01/frog/zhengzhigao/feluca_exp/example/sample

#small example with output
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/output.vertices ./example/output.edges 6 8 3 2 4 >file.txt  2>&1
#without output
./vpublic01/frog/zhengzhigao/feluca_exp/exp ${datapath}/output.vertices ${datapath}/output.edges 6 8 3 2 4  ${datapath}/sample_graph.txt