
//#pragma once 
#include <stdio.h>
#include "graph.h"

/*
Name: CHECK
Copyright: 
Author: Xuan Luo 
Date: 14/01/16 16:50
Description:check whether the value is right or not 
*/


/* print the value of array g */
void checkvalue_s(int * g, int size)
{
	for(int i=0;i<size;i++)
		printf("%d\t",g[i]);
	printf("\n");
}

/* print the value of array g and m*/
void checkvalue_d(int * g, int *m,int size)
{
	for(int i=0;i<size;i++)
		printf("%d ----> %d \t",g[i],m[i]);
	printf("\n");
}

/* check graph data structure */
void checkGraphvalue(Graph ** g, DataSize * size,int gpu_num)
{
	int edge_local_num;
	for(int i=0;i<gpu_num;i++)
	{
		printf("******GPU %d Information**********\n",i);
		/*  vertice */
		printf("vertex_num: %d\n",g[i]->vertex_num);
		printf("vertex_duplicate_num: %d\n", g[i]->vertex_duplicate_num);
		printf("vertex_id:\n");
		checkvalue_s(g[i]->vertex_id,g[i]->vertex_num);
		printf("vertex_duplicate_id:\n");
		checkvalue_s(g[i]->vertex_duplicate_id,g[i]->vertex_duplicate_num);

        /* edges */
		printf("edge_num: %d\n",g[i]->edge_num);
		printf("edge_duplicate_num: %d\n",g[i]->edge_duplicate_num); 
        printf("edge list of duplicate:\n");
        checkvalue_d(g[i]->edge_duplicate_src,g[i]->edge_duplicate_dst,g[i]->edge_duplicate_num);
        printf("edge list of local:\n");
        edge_local_num=g[i]->edge_num-g[i]->edge_duplicate_num;
        checkvalue_d(g[i]->edge_local_src,g[i]->edge_local_dst,edge_local_num);
	}
}

void checkResult(int *g, int *m, int num)
{
   int i=0;
   for (i = 0; i < num; ++i)
   {
   	  if (g[i]!=m[i])
   	  {
   	  	 break;
   	  }
   }
   if (i<num)
   {
   	  printf("Check Fail!\n");
   }
   else
   	printf("Check Success!\n");

}