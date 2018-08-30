git pull
rm -rf ./src/*.o  ./exp  
rm -rf obj
make clean
make

datapath=/vpublic01/frog/zhengzhigao/feluca4big/example

./vpublic01/frog/zhengzhigao/feluca_exp/exp ${datapath}/rmat/rmat.vertices ${datapath}/rmat/rmat.edges 735292 3523247 539587 1761624 4

#datapath=/home/xxling/dataset/VGP-master/dist
#origin=/home/xxling/dataset
#backfix=.txt_totem


#rmat16_1
#echo "rmat1..2" >> ./result/ngpurmat1.txt
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT16_1-2.vertices ${datapath}/RMAT16_1-2.edges 1000000 16000000 477204 7985768 2 ${origin}/RMAT16_1${backfix} 2>&1 >> ./result/ngpurmat1.txt


#echo "rmat1..3"  >>  ./result/ngpurmat1.txt
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT16_1-3.vertices ${datapath}/RMAT16_1-3.edges 1000000 16000000  417198 5323846 3 ${origin}/RMAT16_1${backfix} 2>&1 >> ./result/ngpurmat1.txt

#echo "rmat1..4" >> ./result/ngpurmat1.txt 
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT16_1-4.vertices ${datapath}/RMAT16_1-4.edges 1000000 16000000 375070 3992884 4 ${origin}/RMAT16_1${backfix}  >> ./result/ngpurmat1.txt

#rmat16_2
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT6_2.vertices ${datapath}/RMAT6_2.edges 10000000 160000000 
