//Todo:time record on GPU


//#define PRINT_CHECK
//header file of memset()
#include<string.h>
#include<malloc.h>
#include<stdio.h>
#include<omp.h>

#include "graph.h"
#include "timer.h"
//#include "algorithm.h"
#include "cuda_runtime.h"

// The number of partitioning the duplicate chunk must be greater or equal to 1
#define ITERATE_IN_DUPLICATE 1
#define NUM_THREADS 1

#ifdef __CUDA_RUNTIME_H__
#define HANDLE_ERROR(err) if (err != cudaSuccess) {	\
	printf("CUDA Error in %s at line %d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
	exit(1);\
}
#endif  // #ifdef __CUDA_RUNTIME_H__    


void bfs_cpu(Graph_cpu *g,int *value_cpu,DataSize *dsize,int first_vertex)
{
	printf("BFS is running on CPU...............\n");
	timer_start();
	int vertex_num=dsize->vertex_num;
	int edge_num=dsize->edge_num;
	int edge_src,edge_dst;
	int *queue=(int *)malloc(sizeof(int)*vertex_num);
	memset(value_cpu,0,vertex_num*sizeof(int));
	value_cpu[first_vertex]=1;
	if(queue==NULL)
	{
		perror("Out of memory");
		exit(1);
	}

	int step=1;
	int incount=0;
	int outcount=0;
	queue[incount++]=first_vertex;

	while(incount > outcount)
	{
		int vertex_id=queue[outcount++];
		for (int i = g->vertex_begin[vertex_id]; i < g->vertex_begin[vertex_id+1]; ++i)
		{
			int dst_id=g->vertex_dst[i];
			step=value_cpu[vertex_id];
			if (value_cpu[dst_id]==0)
			{
				value_cpu[dst_id]=step+1;
				queue[incount++]=dst_id;
			}
		}

#ifdef PRINT_CHECK
		printf("\n");
		for (int i = 0; i < 15 && i<vertex_num+1; ++i)
		{
			printf("%d\t", value_cpu[i]);
		}
		printf("\n");
#endif

	}
	double total_time=timer_stop();
	printf("Total time of bfs_cpu is %.3fms\n",total_time);
}

// print info about bfs values
void print_bfs_values(const int * const values, int const size) {
	int visited = 0;
	int step = 0;
	int first = 0;

	// get the max step and count the visited
	for (int i = 0; i < size; i++) {
		if (values[i] != 0) {
			visited++;
			if (values[i] > step) step = values[i];
			if (values[i] == 1) first = i;
		}
	}
	// count vertices of each step
	if (step == 0) return;
	int * m = (int *) malloc((step + 1)*sizeof(int));
	memset(m,0,sizeof(int)*(step+1));
	for (int i = 0; i < size; i++) {
		m[values[i]]++;
	}
	// print result info
	printf("\tSource = %d, Step = %d, Visited = %d\n", first, step, visited);
	printf("\tstep\tvisit\n");
	for (int i = 1; i <= step; i++) {
		printf("\t%d\t%d\n", i, m[i]);
	}
	printf("\n");
	free(m);
}

static __global__ void  bfs_kernel_duplicate(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		const int step)
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// step counter
	int curStep = step;
	int nextStep = curStep + 1;
	// proceeding loop
	for (int i = index; i < edge_num; i +=n) {		
		if (values[edge_src[i]] == curStep && values[edge_dest[i]] == 0) {
			values[edge_dest[i]] = nextStep;
		}
	}
}
static __global__ void bfs_kernel_local(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		const int step,
		int * const continue_flag)
{

	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	int curStep = step;
	int nextStep = curStep + 1;

	for (int i = index; i < edge_num; i +=n) {		
		if(values[edge_src[i]]==curStep && values[edge_dest[i]]==0)	
		{
			values[edge_dest[i]]=nextStep;
			flag = 1;
		}	
	}
	// update flag
	if (flag == 1) *continue_flag = 1;
}
static __global__ void kernel_make_bitmap(
		int const vertex_num,
		int const * const values,
		int * const bitmap,
		int const targe)
{
	int const n=blockDim.x*gridDim.x;
	int const tid=threadIdx.x+blockDim.x*blockIdx.x;
	for (int i = tid; i < vertex_num; i+=n)
	{
		int const v=__ballot(values[i]==targe);
		bitmap[i>>5]=v;
	}
}

static __global__ void  kernel_extract_bitmap(  
		int const vertex_num,
		int const * const bitmap,
		int * const values,
		int const targe)
{
	int const n=blockDim.x*gridDim.x;
	int const tid=threadIdx.x+blockIdx.x*blockDim.x;
	int const mask=1<<(tid & 31);
	for (int i = tid; i < vertex_num; i=i+n)
	{
		if(bitmap[i>>5]&mask) values[i]=targe;
	}
}


void merge_bitmap_on_cpu(
		int const bitmap_len,
		int const gpu_num,
		int * const *  bitmap,
		int * const  buffer,
		int &flag)
{
	int i,id;
	omp_set_num_threads(NUM_THREADS);	
#pragma omp parallel private(i)
	{
		id=omp_get_thread_num(); 
		for (i = id; i < bitmap_len; i=i+NUM_THREADS)
		{
			buffer[i]=0;
			int t=0;
			for (int j = 0; j < gpu_num; ++j)
			{
				t=t| bitmap[j][i];
				if(t) 
				{
					buffer[i]=t;
					flag=1;
					break;
				}
			}
		}

	}
}

void merge_bitmap_on_cpu_1(
		int const bitmap_len,
		int const duplicate_num,
		int const gpu_num,
		int * const *  bitmap,
		int * const  buffer,
		int &flag)
{
	int i,id;
	{
		for (i = 0; i < bitmap_len*duplicate_num; i++)
		{
			int t=0;
			for (int j = 0; j < gpu_num; ++j)
			{
				t=t| bitmap[j][i];

			}
			bitmap[0][i]=t;
		}
		for(i=0; i< bitmap_len;i++)
		{
			int t=0;
			buffer[i]=0;
			for(int j=0; j<duplicate_num;j++)
			{
				int *p=bitmap[0]+bitmap_len*j;
				t=t|p[i];
				if(t)
				{
					buffer[i]=t;
flag=1;
					break;
				}
			}

		}

	}
}

void Gather_result(
		int vertex_num,
		int gpu_num,
		int * const * const h_value,
		int * const value_gpu)
{
	omp_set_num_threads(NUM_THREADS);
	int j,id;	
#pragma omp parallel private(j)
	for (int i = 0; i < gpu_num; ++i)
	{
		//		int *edge_dest=g[i]->edge_local_dst;
		//	int size=g[i]->edge_num-g[i]->edge_duplicate_num;
		id=omp_get_thread_num(); 
		for (j = id; j <vertex_num	; j=j+NUM_THREADS)
		{
			if(h_value[i][j]>0)
				value_gpu[j]=h_value[i][j];
		}
	}
}
/* BFS algorithm on GPU */
void bfs_gpu(Graph **g,int gpu_num,int *value_gpu,DataSize *dsize, int first_vertex, int *copy_num, int **position_id)
{
	printf("BFS is running on GPU...............\n");
	printf("Start malloc edgelist...\n");
	/* TODO : edgelsit store twices */
	/* Inite value*/
	value_gpu[first_vertex]=1;
	// TODO : can be deleted
	int **h_value=(int **)malloc(sizeof(int *)* gpu_num);
	int **h_flag=(int **)malloc(sizeof(int *)*gpu_num);
	int vertex_num=dsize->vertex_num;
	int **d_edge_local_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_local_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_duplicate_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_duplicate_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_value=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_flag=(int **)malloc(sizeof(int *)*gpu_num);

	//add
	int  bitmap_len=(vertex_num+sizeof(int)*8-1)/(sizeof(int)*8);
	int  **h_bitmap=(int **)malloc(sizeof(int *)*gpu_num); 
	int  **d_bitmap=(int **)malloc(sizeof(int *)*gpu_num); 
	int *buff_bitmap=(int *)malloc(sizeof(int)*bitmap_len);


	/* determine the size of duplicate vertex in one process*/
	int tmp_per_size = min_num_duplicate_edge(g,gpu_num);
	int duplicate_per_size=tmp_per_size/ITERATE_IN_DUPLICATE;
	int iterate_in_duplicate=ITERATE_IN_DUPLICATE+1;
	int *last_duplicate_per_size=(int *)malloc(sizeof(int)*gpu_num);
	memset(last_duplicate_per_size,0,sizeof(int)*gpu_num);



	for (int i = 0; i < gpu_num; ++i)
	{
		h_value[i]=(int *)malloc(sizeof(int)*(vertex_num+1));
		memset(h_value[i],0,sizeof(int)*(vertex_num+1));
		h_value[i][first_vertex]=1;
		h_flag[i]=(int *)malloc(sizeof(int));

		//add
		h_bitmap[i]=(int *)malloc(sizeof(int)*(bitmap_len*iterate_in_duplicate));
		memset(h_bitmap[i],0,sizeof(int)*(bitmap_len*iterate_in_duplicate));
	}



	/*Cuda Malloc*/
	/* Malloc stream*/
	cudaStream_t **stream;
	cudaEvent_t tmp_start,tmp_stop;
	stream=(cudaStream_t **)malloc(gpu_num*sizeof(cudaStream_t*));

	cudaEvent_t * start_duplicate,*stop_duplicate,*start_local,*stop_local,*start_asyn,*stop_asyn,*start,*stop;
	start_duplicate=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_duplicate=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_local=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_local=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));

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
		HANDLE_ERROR(cudaEventCreate(&start[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop[i],0));
		HANDLE_ERROR(cudaEventCreate(&tmp_start,0));
		HANDLE_ERROR(cudaEventCreate(&tmp_stop,0));

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
		HANDLE_ERROR(cudaMalloc((void **)&d_bitmap[i],sizeof(int)*(bitmap_len*iterate_in_duplicate)));

		if (duplicate_per_size!=0 && duplicate_per_size < out_size)
		{
			for (int j = 1; j < iterate_in_duplicate; ++j)
			{
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_src[i]+(j-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_src+(j-1)*duplicate_per_size),sizeof(int)*duplicate_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_dst[i]+(j-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_dst+(j-1)*duplicate_per_size),sizeof(int)*duplicate_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_bitmap[i]+(j-1)*bitmap_len),(void *)(h_bitmap[i]+(j-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyHostToDevice,stream[i][j-1]));
			}
		}

		last_duplicate_per_size[i]=g[i]->edge_duplicate_num-duplicate_per_size * (iterate_in_duplicate-1);           
		if (last_duplicate_per_size[i]>0 && iterate_in_duplicate>1 )
		{
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_src[i]+(iterate_in_duplicate-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_src+(iterate_in_duplicate-1)*duplicate_per_size),sizeof(int)*last_duplicate_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_duplicate-1]));
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_duplicate_dst[i]+(iterate_in_duplicate-1)*duplicate_per_size),(void *)(g[i]->edge_duplicate_dst+(iterate_in_duplicate-1)*duplicate_per_size),sizeof(int)*last_duplicate_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_duplicate-1]));
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),(void *)(h_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),sizeof(int)*bitmap_len,cudaMemcpyHostToDevice,stream[i][iterate_in_duplicate-1]));
		}


		HANDLE_ERROR(cudaMalloc((void **)&d_edge_local_src[i],sizeof(int)*local_size));
		HANDLE_ERROR(cudaMalloc((void **)&d_edge_local_dst[i],sizeof(int)*local_size));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_local_src[i],(void *)g[i]->edge_local_src,sizeof(int)*local_size,cudaMemcpyHostToDevice,stream[i][iterate_in_duplicate]));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_local_dst[i],(void *)g[i]->edge_local_dst,sizeof(int)*local_size,cudaMemcpyHostToDevice,stream[i][iterate_in_duplicate]));

		HANDLE_ERROR(cudaMalloc((void **)&d_value[i],sizeof(int)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_value[i],(void *)h_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));

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
	int flag=0;
	int step=1;
	int local_edge_num=0;

#ifdef PRINT_CHECK
	for (int i = 0; i < gpu_num; ++i)
	{

		HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
		printf("Before while --> check value\n");
		printf("value:\n");
		for (int j = 0; j < vertex_num+1; ++j)
		{
			printf("%d\t", h_value[i][j]);
		}
		printf("\nDuplicate_edgelist\n");
		for (int j = 0; j < g[i]->edge_duplicate_num ; ++j)
		{
			printf("( %d, %d )\t",g[i]->edge_duplicate_src[j],g[i]->edge_duplicate_dst[j]);
		}
		printf("\nlocal_edgelist\n");
		for (int j = 0; j < g[i]->edge_num- g[i]->edge_duplicate_num; ++j)
		{
			printf("( %d, %d )\t",g[i]->edge_local_src[j],g[i]->edge_local_dst[j]);
		}
		printf("\n");
	}
#endif

	/* one iteration */
	printf("Computing......\n");
	do
	{
		flag=0;
		for (int i = 0; i <gpu_num; ++i)
		{		
			memset(h_flag[i],0,sizeof(int));
			cudaSetDevice(i);
            HANDLE_ERROR(cudaMemset(d_bitmap[i],0,sizeof(int)*(bitmap_len*iterate_in_duplicate)));	
			HANDLE_ERROR(cudaMemset(d_flag[i],0,sizeof(int)));

			HANDLE_ERROR(cudaEventRecord(start_duplicate[i], stream[i][0]));
			//kernel of duplicate edgelist
			if (duplicate_per_size!=0 && duplicate_per_size < g[i]->edge_duplicate_num)
			{
				for (int j = 1; j < iterate_in_duplicate; ++j)
				{				
					bfs_kernel_duplicate<<<208,128,0,stream[i][j-1]>>>(
							duplicate_per_size,
							d_edge_duplicate_src[i]+(j-1)*duplicate_per_size,
							d_edge_duplicate_dst[i]+(j-1)*duplicate_per_size,
							d_value[i],
							step);
					kernel_make_bitmap<<<208,128,0,stream[i][j-1]>>>(
							vertex_num,
							d_value[i],
							d_bitmap[i]+(j-1)*bitmap_len,
							(step+1));
					//HANDLE_ERROR(cudaMemcpyAsync((void *)(h_bitmap[i]+(j-1)*bitmap_len),(void *)(d_bitmap[i]+(j-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost,stream[i][j-1]));
					//HANDLE_ERROR(cudaMemcpy((void *)(h_bitmap[i]+(j-1)*bitmap_len),(void *)(d_bitmap[i]+(j-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost,stream[i][j-1]));
				}
			}

			last_duplicate_per_size[i]=g[i]->edge_duplicate_num-duplicate_per_size * (iterate_in_duplicate-1);           
			if (last_duplicate_per_size[i]>0 && iterate_in_duplicate>1  )
			{
				// The size of edge list in last block is different in every gpu
				bfs_kernel_duplicate<<<208,128,0,stream[i][iterate_in_duplicate-1]>>>(
						last_duplicate_per_size[i],
						d_edge_duplicate_src[i]+(iterate_in_duplicate-1)*duplicate_per_size,
						d_edge_duplicate_dst[i]+(iterate_in_duplicate-1)*duplicate_per_size,
						d_value[i],
						step);
				kernel_make_bitmap<<<208,128,0,stream[i][iterate_in_duplicate-1]>>>(
						vertex_num,
						d_value[i],
						d_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len,
						step+1
						);
				//HANDLE_ERROR(cudaMemcpyAsync((void *)(h_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),(void *)(d_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost,stream[i][iterate_in_duplicate-1]));
			   //HANDLE_ERROR(cudaMemcpy((void *)(h_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),(void *)(d_bitmap[i]+(iterate_in_duplicate-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost));
			}
			HANDLE_ERROR(cudaEventRecord(stop_duplicate[i], stream[i][iterate_in_duplicate-1]));
            
           HANDLE_ERROR(cudaMemcpy(h_bitmap[i],d_bitmap[i],sizeof(int)*(bitmap_len*iterate_in_duplicate),cudaMemcpyDeviceToHost));			
#ifdef PRINT_CHECK_1
			printf("The value after bfs_duplicate_kernel\n");
			HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_bitmap[i],d_bitmap[i],sizeof(int)*(bitmap_len*iterate_in_duplicate),cudaMemcpyDeviceToHost));
			printf("@@value\n");
			for (int j = 0; j < vertex_num+1 && j<10; ++j)
			{
				printf("%d\t", h_value[i][j]);
			}
			printf("\n@@bitmap:\n");
			for(int j=0;j<bitmap_len*iterate_in_duplicate;j++)
			{
				printf("%d\t",h_bitmap[i][j]);
			}
			printf("\n\n");
#endif

			HANDLE_ERROR(cudaEventRecord(start_local[i], stream[i][iterate_in_duplicate]));
			//local+flag
			local_edge_num=g[i]->edge_num-g[i]->edge_duplicate_num;
			if (local_edge_num>0)
			{
				bfs_kernel_local<<<208,128,0,stream[i][iterate_in_duplicate]>>>(
						local_edge_num,
						d_edge_local_src[i],
						d_edge_local_dst[i],
						d_value[i],
						step,
						d_flag[i]);			
				HANDLE_ERROR(cudaMemcpyAsync(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost,stream[i][iterate_in_duplicate]));	    
			}
			HANDLE_ERROR(cudaEventRecord(stop_local[i],stream[i][iterate_in_duplicate]));


#ifdef PRINT_CHECK_1
			printf("The value after bfs_local_kernel\n");
			HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_bitmap[i],d_bitmap[i],sizeof(int)*(bitmap_len*iterate_in_duplicate),cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost));
			printf("@@ value\n");
			for (int j = 0; j < vertex_num+1 && j<10; ++j)
			{
				printf("%d\t", h_value[i][j]);
			}
			printf("\n @@GPU flag:%d\n",h_flag[i][0]);
			printf("bitmap:\n");
			for(int j=0;j<bitmap_len*iterate_in_duplicate;j++)
			{
				printf("%d\t",h_bitmap[i][j]);
			}
			printf("\n\n");
#endif

		}


		//merge bitmap on gpu
		double t1=omp_get_wtime();
		merge_bitmap_on_cpu_1(bitmap_len, iterate_in_duplicate, gpu_num, h_bitmap, buff_bitmap,flag);
		double t2=omp_get_wtime();
		record_time=(t2-t1)*1000;
		gather_time+=record_time;


#ifdef PRINT_CHECK_1
		printf("-----------------------------After merge\n");
		printf("value:\n");
		for (int i = 0; i < gpu_num; ++i)
		{
			HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			for (int j = 0; j < vertex_num+1 && j<10; ++j)
			{
				printf("%d\t", h_value[i][j]);
			}
			printf("\n");

		}
        
		printf("@@bitmap:\n");
	for (int i = 0; i < bitmap_len; ++i)
	{
		printf("%d\n",buff_bitmap[i]);
	}
		printf("@@ flag %d\n\n", flag);


#endif


		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			//extract bitmap to the value
			HANDLE_ERROR(cudaMemcpyAsync(d_bitmap[i], buff_bitmap,sizeof(int)*bitmap_len,cudaMemcpyHostToDevice,stream[i][0]));
			HANDLE_ERROR(cudaEventRecord(start_asyn[i], stream[i][0]));
			kernel_extract_bitmap<<<256,108,0,stream[i][0]>>>
				(  
				 vertex_num,
				 d_bitmap[i],
				 d_value[i],
				 step+1
				);		
			HANDLE_ERROR(cudaEventRecord(stop_asyn[i], stream[i][0]));
			HANDLE_ERROR(cudaMemset(d_bitmap[i],0,sizeof(int)*(bitmap_len*iterate_in_duplicate)));	
		}

#ifdef PRINT_CHECK_1
		//HANDLE_ERROR(cudaMemcpy(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost));
		printf("-----------------------------After extract\n");
		printf("value\n");
		for (int i = 0; i < gpu_num; ++i)
		{
			HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			for (int j = 0; j < vertex_num+1 && j<10; ++j)
			{
				printf("%d\t", h_value[i][j]);

			}
			printf("\n\n");
		}


#endif


#ifdef PRINT_CHECK_1
		printf("The value after bfs_extral_bitmap/   before next iteration\n");
		printf("buff_bitmap\n");
		for(int j=0 ; j<bitmap_len;j++)
			printf("%d\t",buff_bitmap[j]);
		printf("\nvalue\n");
		for(int i=0;i<gpu_num;i++)
		{
			HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
			for (int j = 0; j < vertex_num+1 && j<10; ++j)
			{
				printf("%d\t", h_value[i][j]);
			}
			printf("\n");
		}
		printf("\nCPU flag:%d\n",flag);
#endif


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
	}while(flag && step<1000);


	//Todo to get the true value of local vertice and duplicate vertice
	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync((void *)h_value[i],(void *)d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][0]);
	}

	printf("Gather result on cpu....\n");
	Gather_result(vertex_num+1,gpu_num,h_value,value_gpu);

	printf("Time print\n");

	//collect the information of time 
	float total_time_n=0.0;
	for (int i = 0; i < gpu_num; ++i)
	{
		if(total_time_n<total_compute_time[i])
			total_time_n=total_compute_time[i];
	}
	total_time=total_time_n>gather_time?total_time_n:gather_time;

//	printf("Total time of bfs_gpu is %.3f ms\n",total_time);
	printf("Elapsed time of bfs_gpu is %.3f ms\n", total_time/step);
	printf("%d step\n",step);
	printf("-------------------------------------------------------\n");
	printf("Detail:\n");
	printf("\n");
	for (int i = 0; i < gpu_num; ++i)
	{
		printf("GPU %d\n",i);
		printf("Duplicate_Compute_Time(include pre-stage):  %.3f ms\n", duplicate_compute_time[i]/step);
		printf("local_Compute_Time:                     %.3f ms\n", local_compute_time[i]/step);
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
		HANDLE_ERROR(cudaEventDestroy(start[i]));
		HANDLE_ERROR(cudaEventDestroy(stop[i]));
		HANDLE_ERROR(cudaFree(d_edge_duplicate_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_duplicate_dst[i]));
		HANDLE_ERROR(cudaFree(d_edge_local_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_local_dst[i]));
		HANDLE_ERROR(cudaFree(d_value[i]));
		HANDLE_ERROR(cudaFree(d_flag[i]));

		HANDLE_ERROR(cudaDeviceReset());
		//error 
		//free(h_value[i]);
		free(h_flag[i]);
		free(stream[i]);
	}

	free(duplicate_compute_time);
	free(local_compute_time);
	free(compute_time);
}