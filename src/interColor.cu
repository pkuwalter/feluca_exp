#include<malloc.h>
#include<stdio.h>
#include<omp.h>
#include <thrust/count.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <thrust/count.h>
#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <helper_cuda.h>

#include "graph.h"
#include "timer.h"
//#include "algorithm.h"
#include "cuda_runtime.h"


// The number of partitioning the duplicate chunk must be greater or equal to 1
#define ITERATE_IN_DUPLICATE 2
#define NUM_THREADS 1

#define PAGERANK_COEFFICIENT  0.85f
#define PAGERANK_THRESHOLD  0.005f

#ifdef __CUDA_RUNTIME_H__
#define HANDLE_ERROR(err) if (err != cudaSuccess) {	\
	printf("CUDA Error in %s at line %d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
	exit(1);\
}
#endif  // #ifdef __CUDA_RUNTIME_H__    

static __global__ void  coloring_kernel_duplicate(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		//const int * const out_degree,
		int * const colors,
		int * const un_colored)
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = index; i < edge_num; i+=n)
	{
		int src=edge_src[i];
		int dest=edge_dest[i];
		if(colors[src] == colors[dest])
		{
			colors[dest] = colors[src] + 1;
			un_colored[dest] = 1;
		}
	}
}

static __global__ void coloring_kernel_local(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		//const int * const out_degree,
		int * const colors,
		int * const d_uncolored,
		int * const continue_flag)
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = index; i < edge_num; i+=n)
	{
		int src=edge_src[i];
		int dest=edge_dest[i];
		if (colors[src] == colors[dest])
		{
			colors[dest] = colors[src] + 1;
			d_uncolored[dest] = 1;
		}		
	}
	__syncthreads();	
}


static __global__ void kernel_extract_color(
		int const edge_num,
		int * const edge_dest,
		int * const add_color,
		int * const colors
		)
{
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = index; i < edge_num; i+=n)
	{
		int dest=edge_dest[i];
		colors[dest]=add_color[dest];
		add_color[dest]=1;
	} 
}

void merge_colors_on_cpu(
		int const vertex_num, 
		int const gpu_num, 
		int * const * h_add_color, 
		int * const color_gpu, 
		int *copy_num, 
		int *uncolored,
		int flag)
{
	int i,id;
	float new_color=0.0f;
	omp_set_num_threads(NUM_THREADS);	
#pragma omp parallel private(i)
	{
		id=omp_get_thread_num(); 
		for (i = id; i < vertex_num; i=i+NUM_THREADS)
		{
			if (copy_num[i]>1)
			{
				
				for (int j = 0; j < gpu_num; ++j)
				{
					color_gpu[i] = h_add_color[j][i];  
				}

				/**************************
				int num_undone = std::count(uncolored, uncolored + vertex_num, 1);
				
				float percentage = 0.05;
				if(percentage < (float)num_undone/vertex_num)
					break;
				**********************/

				/********************
				new_color=0.0f;
				for (int j = 0; j < gpu_num; ++j)
				{
					new_color+=h_add_color[j][i];  
				}
				new_color=PAGERANK_COEFFICIENT*new_color+1.0 - PAGERANK_COEFFICIENT;
				if(fabs(new_color- color_gpu[i]>PAGERANK_THRESHOLD))
					//flag=1;
				color_gpu[i]=new_color;
				**************************/
			}
		}
	}
}

void Gather_result_colors(
		int const vertex_num, 
		int const gpu_num, 
		int * const copy_num,
		int * const  *h_add_color,  
		int * const color_gpu
		)
{
	int i,id;
	float new_color=0.0f;
	omp_set_num_threads(NUM_THREADS);	
#pragma omp parallel private(i)
	{
		id=omp_get_thread_num(); 
		for (i = id; i < vertex_num; i=i+NUM_THREADS)
		{
			if (copy_num[i]>1)
			{
				for (int j = 0; j < gpu_num; ++j)
				{
					color_gpu[i] = h_add_color[j][i];  
				}
				
				/*******************
				new_color=0.0f;
				for (int j = 0; j < gpu_num; ++j)
				{
					new_color+=h_add_color[j][i];  
				}
				new_color=PAGERANK_COEFFICIENT*new_color+1.0 - PAGERANK_COEFFICIENT;
				color_gpu[i]=new_color;	
				*******************/
			}
		}
	}
}

/* GraphColoring algorithm on GPU */
void coloring_gpu(Graph **g,int gpu_num,int *color_gpu,DataSize *dsize, int* out_degree, int *copy_num, int **position_id)
{
	printf("PageRank is running on GPU...............\n");
	printf("Start malloc edgelist...\n");

	int init_num_colors = 100;
	int **h_flag=(int **)malloc(sizeof(int *)*gpu_num);
	int vertex_num=dsize->vertex_num;
	int **d_edge_local_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_local_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_duplicate_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_duplicate_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **h_color=(int **)malloc(sizeof(int *)* gpu_num);
	int **h_add_color=(int **)malloc(sizeof(int *)*gpu_num);
	int *uncolored = (int *)malloc(sizeof(int) * vertex_num);

	int **d_color=(int **)malloc(sizeof(int *)*gpu_num);
	
	//float **d_tem_value=(float **)malloc(sizeof(float *)*gpu_num);
	int **d_add_color=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_outdegree=(int **)malloc(sizeof(int *)*gpu_num);

	int **d_flag=(int **)malloc(sizeof(int *)*gpu_num);

	/* determine the size of duplicate vertex in one process*/
	int tmp_per_size = min_num_duplicate_edge(g,gpu_num);
	int duplicate_per_size=tmp_per_size/ITERATE_IN_DUPLICATE;
	int iterate_in_duplicate=ITERATE_IN_DUPLICATE+1;
	int *last_duplicate_per_size=(int *)malloc(sizeof(int)*gpu_num);
	memset(last_duplicate_per_size,0,sizeof(int)*gpu_num);

	for (int i = 0; i < gpu_num; ++i)
	{
		h_color[i]=(int *)malloc(sizeof(int)*(vertex_num+1));
		h_add_color[i]=(int *)malloc(sizeof(int)*(vertex_num+1));
		//初始化颜色值 
		memset(h_color[i],rand() % init_num_colors,sizeof(int)*(vertex_num+1));

		printf("The Initialization Color is as follow\n");
		printf("GPU Num\tColor\n");

		for(int j =0; j < vertex_num; j++)
		{
			printf("%d\t%d\n",i,h_color[i][j]);
		}

		h_flag[i]=(int *)malloc(sizeof(int));
	}

	/*Cuda Malloc*/
	/* Malloc stream*/
	cudaStream_t **stream;
	cudaEvent_t tmp_start,tmp_stop;
	stream=(cudaStream_t **)malloc(gpu_num*sizeof(cudaStream_t*));

	cudaEvent_t * start_duplicate,*stop_duplicate,*start_local,*stop_local,*start_asyn,*stop_asyn;
	start_duplicate=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_duplicate=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_local=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_local=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));

	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		stream[i]=(cudaStream_t *)malloc((iterate_in_duplicate+1)*sizeof(cudaStream_t));
		HANDLE_ERROR(cudaEventCreate(&start_duplicate[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_duplicate[i],0));
		HANDLE_ERROR(cudaEventCreate(&start_local[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_local[i],0));  
		HANDLE_ERROR(cudaEventCreate(&start_asyn[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_asyn[i],0));


		for (int j = 0; j <= iterate_in_duplicate; ++j)
		{
			HANDLE_ERROR(cudaStreamCreate(&stream[i][j]));
		}
	}

	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		int out_size=g[i]->edge_duplicate_num;
		int local_size=g[i]->edge_num - out_size;

		HANDLE_ERROR(cudaMalloc((void **)&d_edge_duplicate_src[i],sizeof(int)*out_size));
		HANDLE_ERROR(cudaMalloc((void **)&d_edge_duplicate_dst[i],sizeof(int)*out_size));

		if (duplicate_per_size!=0 && duplicate_per_size < out_size)
		{
			for (int j = 1; j < iterate_in_duplicate; ++j)
			{
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_src[i]+(j-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_src+(j-1)*duplicate_per_size),sizeof(int)*duplicate_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_dst[i]+(j-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_dst+(j-1)*duplicate_per_size),sizeof(int)*duplicate_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));			
			}
		}

		last_duplicate_per_size[i]=g[i]->edge_duplicate_num-duplicate_per_size * (iterate_in_duplicate-1);           
		if (last_duplicate_per_size[i]>0 && iterate_in_duplicate>1 )
		{
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_src[i]+(iterate_in_duplicate-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_src+(iterate_in_duplicate-1)*duplicate_per_size),sizeof(int)*last_duplicate_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_duplicate-1]));
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_dst[i]+(iterate_in_duplicate-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_dst+(iterate_in_duplicate-1)*duplicate_per_size),sizeof(int)*last_duplicate_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_duplicate-1]));
		}


		HANDLE_ERROR(cudaMalloc((void **)&d_edge_local_src[i],sizeof(int)*local_size));
		HANDLE_ERROR(cudaMalloc((void **)&d_edge_local_dst[i],sizeof(int)*local_size));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_local_src[i],(void *)g[i]->edge_local_src,sizeof(int)*local_size,cudaMemcpyHostToDevice,stream[i][iterate_in_duplicate]));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_local_dst[i],(void *)g[i]->edge_local_dst,sizeof(int)*local_size,cudaMemcpyHostToDevice,stream[i][iterate_in_duplicate]));

		HANDLE_ERROR(cudaMalloc((void **)&d_color[i],sizeof(int)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_color[i],(void *)h_color[i],sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));
		//pr different
		HANDLE_ERROR(cudaMalloc((void **)&d_add_color[i],sizeof(int)*(vertex_num+1)));
		//"memset only works for bytes. If you're using the runtime API, you can use thrust::fill() instead"
		//HANDLE_ERROR(cudaMemset((void **)&d_add_color[i],0,sizeof(float)*(vertex_num+1)));

		//HANDLE_ERROR(cudaMalloc((void **)&d_tem_value[i],sizeof(float)*(vertex_num+1)));
		//HANDLE_ERROR(cudaMalloc((void **)&d_tem_value[i],sizeof(float)*(vertex_num+1)));
		HANDLE_ERROR(cudaMalloc((void **)&d_outdegree[i],sizeof(int)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync(d_outdegree[i],out_degree, sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));

		HANDLE_ERROR(cudaMalloc((void **)&d_flag[i],sizeof(int)));


	}
	printf("Malloc is finished!\n");

	/* Before While: Time Initialization */
	float *duplicate_compute_time,*local_compute_time,*compute_time,*total_compute_time,*extract_bitmap_time;
	float gather_time=0.0;
	float cpu_gather_time=0.0;
	float total_time=0.0;
	float record_time=0.0;
	duplicate_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	local_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	compute_time=(float *)malloc(sizeof(float)*gpu_num);
	total_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	extract_bitmap_time=(float *)malloc(sizeof(float)*gpu_num);

	memset(duplicate_compute_time,0,sizeof(float)*gpu_num);
	memset(local_compute_time,0,sizeof(float)*gpu_num);
	memset(compute_time,0,sizeof(float)*gpu_num);


	/* Before While: Variable Initialization */
	int step=0;
	int flag=0;
	int local_edge_num=0;

	printf("Computing......\n");
	do
	{
		flag=0;
		for (int i = 0; i <gpu_num; ++i)
		{		
			memset(h_flag[i],0,sizeof(int));
			cudaSetDevice(i);
            HANDLE_ERROR(cudaMemset(d_flag[i],0,sizeof(int)));
			HANDLE_ERROR(cudaEventRecord(start_duplicate[i], stream[i][0]));
			//kernel of duplicate edgelist
			if (duplicate_per_size!=0 && duplicate_per_size < g[i]->edge_duplicate_num)
			{
				for (int j = 1; j < iterate_in_duplicate; ++j)
				{				
					coloring_kernel_duplicate<<<208,128,0,stream[i][j-1]>>>(
							duplicate_per_size,
							d_edge_duplicate_src[i]+(j-1)*duplicate_per_size,
							d_edge_duplicate_dst[i]+(j-1)*duplicate_per_size,
							//[i],
							d_color[i],
							d_add_color[i]);
					//TODO didn't not realize overlap
					//HANDLE_ERROR(cudaMemcpyAsync((void *)(h_add_color[i]),(void *)(d_add_color[i]),sizeof(float)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][j-1]));
				}
			}

			last_duplicate_per_size[i]=g[i]->edge_duplicate_num-duplicate_per_size * (iterate_in_duplicate-1);           
			if (last_duplicate_per_size[i]>0 && iterate_in_duplicate>1  )
			{
				coloring_kernel_duplicate<<<208,128,0,stream[i][iterate_in_duplicate-1]>>>(
						last_duplicate_per_size[i],
						d_edge_duplicate_src[i]+(iterate_in_duplicate-1)*duplicate_per_size,
						d_edge_duplicate_dst[i]+(iterate_in_duplicate-1)*duplicate_per_size,
						//d_outdegree[i],
						d_color[i],
						d_add_color[i]);
				//TODO didn't not realize 
				//HANDLE_ERROR(cudaMemcpyAsync((void *)(h_add_color[i]),(void *)(d_add_color[i]),sizeof(float)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][iterate_in_duplicate-1]));
			}
			HANDLE_ERROR(cudaEventRecord(stop_duplicate[i], stream[i][iterate_in_duplicate-1]));

            HANDLE_ERROR(cudaMemcpy((void *)(h_add_color[i]),(void *)(d_add_color[i]),sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaEventRecord(start_local[i], stream[i][iterate_in_duplicate]));
			//local+flag
			local_edge_num=g[i]->edge_num-g[i]->edge_duplicate_num;
			if (local_edge_num>0)
			{
				coloring_kernel_local<<<208,128,0,stream[i][iterate_in_duplicate]>>>(
						local_edge_num,
						d_edge_local_src[i],
						d_edge_local_dst[i],
						//d_outdegree[i],
						d_color[i],
						d_add_color[i],
						d_flag[i]);			
				HANDLE_ERROR(cudaMemcpyAsync(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost,stream[i][iterate_in_duplicate]));	    
			}
			HANDLE_ERROR(cudaEventRecord(stop_local[i],stream[i][iterate_in_duplicate]));
		}


		//merge bitmap on gpu
		double t1=omp_get_wtime();
		merge_colors_on_cpu(vertex_num, gpu_num, h_add_color, color_gpu, copy_num, uncolored, flag);
		double t2=omp_get_wtime();
		record_time=(t2-t1)*1000;
		gather_time+=record_time;


		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			//extract bitmap to the value
			HANDLE_ERROR(cudaMemcpyAsync(d_add_color[i], color_gpu,sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));
			HANDLE_ERROR(cudaEventRecord(start_asyn[i], stream[i][0]));
			// d_color copy to the value of duplicate vertices

			kernel_extract_color<<<208,128,0,stream[i][0]>>>
				(  
				 g[i]->edge_duplicate_num,
				 d_edge_duplicate_dst[i],
				 d_add_color[i],
				 d_color[i]
				);

			HANDLE_ERROR(cudaEventRecord(stop_asyn[i], stream[i][0]));
		}

		for (int i = 0; i < gpu_num; ++i)
		{
			flag=flag||h_flag[i][0];
		}
		step++;

		//collect time  different stream
		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			HANDLE_ERROR(cudaEventSynchronize(stop_duplicate[i]));
			HANDLE_ERROR(cudaEventSynchronize(stop_local[i]));
			HANDLE_ERROR(cudaEventSynchronize(stop_asyn[i]));

			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_duplicate[i], stop_duplicate[i]));
			duplicate_compute_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_local[i], stop_local[i]));  
			local_compute_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_asyn[i], stop_asyn[i]));  
			extract_bitmap_time[i]+=record_time;
			total_compute_time[i]=duplicate_compute_time[i]+extract_bitmap_time[i]-local_compute_time[i]>0?(duplicate_compute_time[i]+extract_bitmap_time[i]):local_compute_time[i];
		}		
	}while(flag && step<10);




	//Todo to get the true value of local vertice and duplicate vertice
	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync((void *)h_color[i],(void *)d_color[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][0]);
	}

	printf("Gather result on cpu....\n");
	Gather_result_colors(vertex_num,gpu_num,copy_num,h_add_color,color_gpu);

	printf("Time print\n");

	//collect the information of time 
	float total_time_n=0.0;
	for (int i = 0; i < gpu_num; ++i)
	{
		if(total_time_n<total_compute_time[i])
			total_time_n=total_compute_time[i];
	}
	total_time=total_time_n>gather_time?total_time_n:gather_time;

	printf("The color value is as follow\n");
	int color_num[4] = {0,0,0,0};
	int max_color = 0;
	int countcolorbegin = 0;
	int countcolorsecond = 0;
	max_color = color_num[0];
	printf("The GPU number \t The vertices\t The color Value\n");
	for (int i =0; i < gpu_num; i++)
	{

		for(countcolorbegin=0;countcolorbegin<vertex_num;countcolorbegin++)
	    {
	    	printf("%d\t%d\t%d\n",i,countcolorbegin,h_color[i][countcolorbegin]);

	          for(countcolorsecond=countcolorbegin+1;countcolorsecond<vertex_num;countcolorsecond++)
	            
	            if(h_color[i][countcolorsecond]==h_color[i][countcolorbegin])
	                break;
	        if(countcolorsecond==vertex_num)
	        {
	            color_num[i]++;
	        }	       
	    }

	    if(color_num[i]>max_color)
		{
			max_color=color_num[i];
		}
	}
	

	printf("The total color is: %d\n",max_color);


//	printf("Total time of pr_gpu is %.3f ms\n",total_time);
	printf("Elapsed time of pr_gpu is %.3f ms\n", total_time/(step));
	printf("-------------------------------------------------------\n");
	printf("Detail:\n");
	printf("\n");
	for (int i = 0; i < gpu_num; ++i)
	{
		printf("GPU %d\n",i);
		printf("Duplicate_Compute_Time(include pre-stage):  %.3f ms\n", duplicate_compute_time[i]/step);
		printf("Local_Compute_Time:                     %.3f ms\n", local_compute_time[i]/step);
		printf("Total Compute_Time                      %.3f ms\n", total_compute_time[i]/step);
		printf("Extract_Bitmap_Time                     %.3f ms\n", extract_bitmap_time[i]/step);
	}
	printf("CPU \n");
	printf("CPU_Gather_Time:                            %.3f ms\n", gather_time/step);
	printf("--------------------------------------------------------\n");

	//clean
	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		//HANDLE_ERROR(cudaEventDestroy(start[i]));
		//HANDLE_ERROR(cudaEventDestroy(stop[i]));
		HANDLE_ERROR(cudaFree(d_edge_duplicate_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_duplicate_dst[i]));
		HANDLE_ERROR(cudaFree(d_edge_local_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_local_dst[i]));
		HANDLE_ERROR(cudaFree(d_color[i]));
		HANDLE_ERROR(cudaFree(d_flag[i]));

		HANDLE_ERROR(cudaDeviceReset());
		//error 
		//free(h_color[i]);
		free(h_flag[i]);
		free(stream[i]);
	}
	free(duplicate_compute_time);
	free(local_compute_time);
	free(compute_time);
}