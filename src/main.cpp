
#include <stdio.h>
#include <malloc.h>
/* The head of strcat()*/

#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "timer.h"
#include "algorithm.h"

using namespace std;
//#include "cuda_runtime.h"
/*
Description: Test the function in the graph.c
*/


int main(int argc, char *argv[])
{

	/* declarations */
	Graph **g;
	/* check the algorithm running on cpu */
	Graph_cpu * origin_g;
	DataSize * dsize;
	/* Record the path to [input].vertices */
	char *filename_v;
	/* Record the path to [input].edges */
	char *filename_e;
	/* Record the origin file name*/
	char *filename_origin;
	int gpu_num;
	/* CPU need to known all the copy number of vertices in the gpus */
	int *copy_num;
	int **position_id;
    int *value_cpu;
    int *value_gpu;
    //pr different
    int *value_gpu_pr;
    int *out_degree;

	if(argc<8)
	{
		printf("Input Command is Error!\n");
		printf("Description:\n");
		//printf("Input filename1 Input filename2 vertex_num  edge_num gpu_num \n ");
		printf("Input filename1 Input filename2 vertex_num  edge_num   max_part_vertex_num  max_part_edge_num  gpu_num\n ");
		//printf("eg: \\home\\xxling\\amazon.vertices \\hofme\\xxling\\amazon.edges 735322 5158012 356275 880813 4\n"); 
		printf("eg: example/amazon/amazon.vertices example/amazon/amazon.edges 735292 3523247 539587 1761624 4\n");
		printf("Note:\n");
		/* TODO print detailed informatition about input command. */
		printf("....to be continued ...\n");
		return 0;        
	}

	/* malloc*/
	dsize=(DataSize *)malloc(sizeof(DataSize));

	filename_v=argv[1];
	filename_e=argv[2];
	filename_origin=argv[2];


	dsize->vertex_num=atoi(argv[3]);
	dsize->edge_num=atoi(argv[4]);
	dsize->max_part_vertex_num=atoi(argv[5]);
	dsize->max_part_edge_num=atoi(argv[6]);
	gpu_num=atoi(argv[7]);

	copy_num=(int *)malloc(sizeof(int)*(dsize->vertex_num));
	memset(copy_num,0,sizeof(int)*(dsize->vertex_num));

    //read two input file to store as edgelist, seperate outer and inner vertex
	g=Initiate_graph (gpu_num,dsize);
	read_graph_vertices(filename_v,g,gpu_num,copy_num);
	read_graph_edges(filename_e,g,gpu_num,copy_num);
  
     
    int edge_num=dsize->edge_num;
    int vertex_num=dsize->vertex_num;
    int first_vertex=3;

    //
   /*
	origin_g=read_graph_edges_again_to_csr(filename_e,edge_num,vertex_num);

	printf("bfs_cpu\n");
	value_cpu=(int *)malloc(sizeof(int)*(vertex_num+1));
	bfs_cpu(origin_g,value_cpu,dsize,first_vertex);
    print_bfs_values(value_cpu,vertex_num+1);
    free(origin_g);
    
    */

    //value_gpu=(int *)malloc(sizeof(int)*(vertex_num+1));
    //bfs_gpu(g,gpu_num,value_gpu,dsize,first_vertex,copy_num,position_id);
    //print_bfs_values(value_gpu,vertex_num+1);
    //free(g);
    //free(value_gpu);

	value_gpu_pr=(int *)malloc(sizeof(int)*(vertex_num+1));
	out_degree=(int *)malloc(sizeof(int)*(vertex_num+1));
    origin_g=read_graph_edges_again_to_csr(filename_origin,edge_num,vertex_num);
    
    get_outdegree(origin_g,vertex_num,out_degree);
    free(origin_g);
    coloring_gpu(g,gpu_num,value_gpu_pr,dsize,out_degree,copy_num,position_id);    


    free(g);
    free(out_degree);


    //checkResult(value_cpu,value_gpu,vertex_num+1);
 
	//free(value_cpu);
	free(dsize);
//	free(copy_num);
//	free(position_id);
	return 0; 
}