#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
    int num_edges = 226800;   // t12_110592     541440
    int num_nodes = 46656;
// Graph structure
struct Graph {
    int num_nodes;
    int num_edges;
    int* neighbors;
    int* indices;
    int* weights;
    int* heights;
    int* excessflows;
};


// Function to create a graph
struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->neighbors = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    graph->indices = (int*)malloc((num_nodes + 1) * sizeof(int));
    graph->weights = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    graph->heights = (int*)calloc(num_nodes, sizeof(int));
    graph->excessflows = (int*)calloc(num_nodes, sizeof(int));
    return graph;
}



// Function to free the graph
void freeGraph(struct Graph* graph) {
    free(graph->neighbors);
    free(graph->indices);
    free(graph->weights);
    free(graph->heights);
    free(graph->excessflows);
    free(graph);
}




// Function to load a graph from a file
struct Graph* loadGraph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return NULL;
    }



    struct Graph* graph = createGraph(num_nodes, num_edges * 2); // Potentially double the edges

    printf("Loading graph from file...\n");

    int* temp_counts = (int*)calloc(num_nodes, sizeof(int));
    int* reverse_counts = (int*)calloc(num_nodes, sizeof(int));

    // First pass: count the number of neighbors for each node
    for (int i = 0; i < num_edges; ++i) {
        int src, dest;
        int weight;
        if (fscanf(file, "%d %d %d\n", &src, &dest, &weight) != 3) {
            printf("Error: Invalid file format.\n");
            fclose(file);
            free(temp_counts);
            free(reverse_counts);
            freeGraph(graph);
            return NULL;
        }
        temp_counts[src]++;
        if (!reverse_counts[dest]) {
            reverse_counts[dest]++;
        }
    }

    // Build the index array
    graph->indices[0] = 0;
    for (int i = 1; i <= num_nodes; ++i) {
        graph->indices[i] = graph->indices[i - 1] + temp_counts[i - 1] + reverse_counts[i - 1];
    }

    // Reset file pointer to the beginning of the file for the second pass
    fseek(file, 0, SEEK_SET);

    int* current_pos = (int*)calloc(num_nodes, sizeof(int));
    int* reverse_pos = (int*)calloc(num_nodes, sizeof(int));

    // Second pass: fill the neighbors array and weights array
    for (int i = 0; i < num_edges; ++i) {
        int src, dest;
        int weight;
        if (fscanf(file, "%d %d %d\n", &src, &dest, &weight) != 3) {
            printf("Error: Invalid file format.\n");
            fclose(file);
            free(temp_counts);
            free(reverse_counts);
            free(current_pos);
            free(reverse_pos);
            freeGraph(graph);
            return NULL;
        }

        int index_src = graph->indices[src] + current_pos[src]++;
        graph->neighbors[index_src] = dest;
        graph->weights[index_src] = weight;

        if (!reverse_counts[dest] || !reverse_pos[dest]) {
            int index_dest = graph->indices[dest] + temp_counts[dest] + reverse_pos[dest]++;
            graph->neighbors[index_dest] = src;
            graph->weights[index_dest] = 0; // Weight 0 for reverse edge
        }
    }

    printf("Graph loaded successfully.\n");

    free(temp_counts);
    free(reverse_counts);
    free(current_pos);
    free(reverse_pos);
    fclose(file);
    return graph;
}







// Function to get the edge index
__host__ __device__ int getEdgeIndex(int* indices,int* neighbors,int src, int dest) {
    for (int i = indices[src]; i < indices[src + 1]; i++) {
        if (neighbors[i] == dest) {
            return i;
        }
    }
    return -1;
}




// Function to initialize preflow
void preflow(struct Graph* graph, int src) {
    graph->heights[src] = graph->num_nodes;
    graph->excessflows[src] = 0;
    for (int i = graph->indices[src]; i < graph->indices[src + 1]; i++) {
        graph->excessflows[src] += graph->weights[i];
    }

    for (int i = graph->indices[src]; i < graph->indices[src + 1]; i++) {
        int neighbor = graph->neighbors[i];
        int capacity = graph->weights[i];

        if (capacity > 0) {
            graph->weights[i] = 0;
            int k = getEdgeIndex(graph->indices,graph->neighbors, neighbor, src);

            if (k != -1) {
                graph->weights[k] += capacity;
            }

            graph->excessflows[neighbor] += capacity;
            graph->excessflows[src] -= capacity;
        }
    }
}





__device__ void relabel(int* indices, int* heights,  int* weights, int* neighbors, int u) {
    int minHeight = INT_MAX;
    for (int i = indices[u]; i < indices[u + 1]; i++) {
        int v = neighbors[i];

        if (weights[i] > 0) {
            if (heights[v] < minHeight) {
                minHeight = heights[v];
            }
      
        }
    }

    if (minHeight < INT_MAX) {
        atomicExch(&heights[u], minHeight + 1);
    }
}



__device__ int push(int* indices, int* heights, int* excessflows, int* weights, int* neighbors, int num_edges, int num_nodes, int u) {
    int f = -1;
    for (int i = indices[u]; i < indices[u + 1]; i++) {
        int v = neighbors[i];

        if (weights[i] > 0 && heights[u] == heights[v] + 1) {
            int delta = min(excessflows[u], weights[i]);

            atomicSub(&weights[i], delta);

            int k = getEdgeIndex(indices,neighbors, v, u); ////
            if (k != -1) {
                atomicAdd(&weights[k], delta);
            }

            atomicSub(&excessflows[u], delta);
            atomicAdd(&excessflows[v], delta);
            f = 0;
            return 0;
        }
    }
    // return f;
    return -1;
}





__global__ void push_relabel_kernel(int* indices, int* heights, int* excessflows, int* weights, int* neighbors, int num_edges, int num_nodes,int src , int dest , int* activenodesarray, int* ccount, int* finalll){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = 0;

    int cycles = 30;
    *finalll = 0;
    
    
    while(cycles!=0){

     for (int i = tid; i < num_nodes; i += stride) {
        if (i != src && i != dest && excessflows[i] > 0 && heights[i] < num_nodes) {
            int index = atomicAdd(ccount, 1);
            activenodesarray[index] = i;
        }
     }
        
    
        
     __syncthreads(); 
     if(*ccount == 0){
         *finalll = 1;
        return;
     }
        
        
        
     size = *ccount;        
        
        
        
        
     for (int i = tid; i < size; i += stride) {
        int u = activenodesarray[i];
        int result = push(indices, heights,  excessflows,  weights, neighbors,  num_edges, num_nodes, u);
        if (result == -1) {
            relabel(indices, heights,  weights, neighbors,  u);    
        }
     }
    
      *ccount = 0;
      cycles--; 
        
        
        
     __syncthreads();    
      
        
    }

}































































__global__ void ddefault(int* lvl,int* heightts, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_nodes; i += stride) {
        lvl[i] = -1;
        heightts[i] = num_nodes;
    }
}



__global__ void bfs_kernel(int* heights, int* excessflows, int num_nodes,int frontier_size, int* indices, int* neighbors, int* weights, int* lvl, int* frontier, int current_level, int* heightts , int* next_frontier_size , int* d_next_frontier) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < frontier_size; i += stride) {
        int u = frontier[i];
        for (int j = indices[u]; j < indices[u + 1]; j++) {
            int v = neighbors[j];
            int weight = weights[getEdgeIndex(indices, neighbors , v, u)];
            if (lvl[v] == -1 && weight > 0 && heightts[v] == num_nodes) {
                int old = atomicCAS(&lvl[v], -1, current_level + 1);
                if (old == -1) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    d_next_frontier[pos] = v;
                    atomicCAS(&heightts[v], num_nodes, heightts[u] + 1);
                }
            }
        }
    }
}





void bfs(int* d_indices, int* d_neighbors,int* d_heights, int* d_excessflows, int* d_weights, int dest, int src, int num_nodes, int num_edges,int* d_lvl,int* d_heightts,int* d_frontier,int* d_next_frontier,bool* d_frontier_visited, int* h_next_frontier, bool* h_frontier_visited) {
    ddefault<<<(num_nodes + 255) / 256, 256>>>(d_lvl, d_heightts, num_nodes);
    int frontier_size = 0;
    int *d_next_frontier_size;
    cudaMalloc(&d_next_frontier_size, sizeof(int));
    int* h_next_frontier_size = (int*)malloc(sizeof(int));
    cudaMemcpy(&d_frontier[frontier_size], &dest, sizeof(int), cudaMemcpyHostToDevice);
    frontier_size++;
    int current_level = 0;
    cudaDeviceSynchronize();
    int value = 0;
    cudaMemcpy(&d_lvl[dest], &value, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_heightts[dest], &value, sizeof(int), cudaMemcpyHostToDevice);
    while (true) {
        cudaMemset(d_next_frontier_size, 0, sizeof(int));
        bfs_kernel<<<1, 1>>>(d_heights, d_excessflows, num_nodes,frontier_size, d_indices, d_neighbors, d_weights, d_lvl, d_frontier, current_level, d_heightts, d_next_frontier_size, d_next_frontier);
        cudaDeviceSynchronize();
        if(frontier_size == 0){
            break;
        }
        cudaMemcpy(h_next_frontier_size,d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_frontier, d_next_frontier, (*h_next_frontier_size) * sizeof(int), cudaMemcpyDeviceToDevice);
        frontier_size = *h_next_frontier_size;
        current_level++;
    }
    cudaMemcpy(d_heights, d_heightts, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&d_heights[src], &num_nodes, sizeof(int), cudaMemcpyHostToDevice);
}


int main(){
    const char* filename = "/content/drive/MyDrive/cuda/t15_46656,226800.txt";
    struct Graph* graph = loadGraph(filename);
    if (!graph) {
        return -1;
    }

    int* d_indices;
    int* d_neighbors;
    int* d_weights;
    int* d_heights;
    int* d_excessflows;
    // int num_edges = 226800;   // t12_110592     541440
    // int num_nodes = 46656;
    cudaMalloc(&d_indices, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_neighbors, num_edges * 2 *sizeof(int));
    cudaMalloc(&d_weights, num_edges * 2 * sizeof(int));
    cudaMalloc(&d_heights, num_edges * 2 * sizeof(int));
    cudaMalloc(&d_excessflows, num_edges * 2 * sizeof(int));


    int* d_lvl;
    cudaMalloc(&d_lvl, num_nodes * sizeof(int));
    int* d_heightts;
    cudaMalloc(&d_heightts, num_nodes * sizeof(int));
    int* d_frontier;
    cudaMalloc(&d_frontier, num_nodes * sizeof(int));
    int* d_next_frontier;
    cudaMalloc(&d_next_frontier, num_nodes * sizeof(int));
    bool* d_frontier_visited;
    cudaMalloc(&d_frontier_visited, num_nodes * sizeof(bool));
    

    int* h_next_frontier = (int*)malloc(num_nodes * sizeof(int));
    bool* h_frontier_visited = (bool*)malloc(num_nodes * sizeof(bool));



















    int src = 0;
    int dest = 9997;

    preflow(graph, src);
  
    

  
    int cc=0;
    
    cudaMemcpy(d_indices, graph->indices, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, graph->neighbors, num_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, graph->weights, num_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heights, graph->heights, (num_nodes) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_excessflows, graph->excessflows, (num_nodes) * sizeof(int), cudaMemcpyHostToDevice);

    











    int* d_activenodesarray;
    cudaMalloc(&d_activenodesarray, graph->num_nodes * sizeof(int));

    int* d_finalll;
    cudaMallocManaged(&d_finalll,sizeof(int));
    int* d_ccount;
    cudaMallocManaged(&d_ccount,sizeof(int));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int *h_finalll = (int *)malloc(sizeof(int));
    int* h_finalexcessflow = (int*)malloc(sizeof(int));
    
    while(true){
        // cudaMemcpy(h_finalexcessflow, &d_excessflows[dest], sizeof(int), cudaMemcpyDeviceToHost);
        //  printf(" , flow = %d\n", *h_finalexcessflow);
        push_relabel_kernel<<<1,1>>>(d_indices, d_heights, d_excessflows,  d_weights,  d_neighbors,num_edges, num_nodes,src,dest,d_activenodesarray,d_ccount,d_finalll);
        
      cudaDeviceSynchronize();
        cudaMemcpy(h_finalll, d_finalll, sizeof(int), cudaMemcpyDeviceToHost);
        if(*h_finalll == 1){
            
            cudaMemcpy(h_finalexcessflow, &d_excessflows[dest], sizeof(int), cudaMemcpyDeviceToHost);
            printf("Max flow from %d to %d is %d\n", src, dest, *h_finalexcessflow);
            break;
        }

         bfs(d_indices, d_neighbors, d_heights, d_excessflows, d_weights,  dest, src, num_nodes, num_edges,d_lvl,d_heightts,d_frontier,d_next_frontier,d_frontier_visited,h_next_frontier,h_frontier_visited);
        
        cc++;
    }

    printf("number of times relabbling occured %d\n",cc);
            clock_gettime(CLOCK_MONOTONIC, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;

    if (nanoseconds < 0) {
        seconds -= 1;
        nanoseconds += 1000000000;
    }

    double elapsed = seconds + nanoseconds * 1e-9;

    printf("Elapsed time: %.9f seconds.\n", elapsed);
    cudaFree(d_indices);
    cudaFree(d_neighbors);
    cudaFree(d_weights);
    freeGraph(graph);
    cudaFree(d_lvl);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_visited);
    cudaFree(d_heightts);
    free(h_next_frontier);
    free(h_frontier_visited);
    
    return 0;



}