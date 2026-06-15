// Smoke-test naive BFS (NOT an experiment candidate) — validates the bfs manifest.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

struct Node { int starting; int no_of_edges; };

__global__ void expand(Node *nodes, int *edges, char *mask, char *upd,
                       char *visited, int *cost, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || !mask[tid]) return;
    mask[tid] = 0;
    int b = nodes[tid].starting, e = b + nodes[tid].no_of_edges;
    for (int i = b; i < e; i++) {
        int nid = edges[i];
        if (!visited[nid]) { cost[nid] = cost[tid] + 1; upd[nid] = 1; }
    }
}
__global__ void commit(char *mask, char *upd, char *visited, int N, int *stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || !upd[tid]) return;
    mask[tid] = 1; visited[tid] = 1; upd[tid] = 0; *stop = 1;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in out\n", argv[0]); return 1; }
    FILE *fp = fopen(argv[1], "r"); if (!fp) return 1;
    int N; fscanf(fp, "%d", &N);
    Node *hn = (Node*)malloc(sizeof(Node)*N);
    char *hmask=(char*)calloc(N,1); int *hcost=(int*)malloc(sizeof(int)*N);
    for (int i=0;i<N;i++){int s,d;fscanf(fp,"%d %d",&s,&d);hn[i].starting=s;hn[i].no_of_edges=d;hcost[i]=-1;}
    int src; fscanf(fp,"%d",&src); src=0;
    int E; fscanf(fp,"%d",&E);
    int *he=(int*)malloc(sizeof(int)*E);
    for(int i=0;i<E;i++){int id,c;fscanf(fp,"%d %d",&id,&c);he[i]=id;}
    fclose(fp);
    hmask[src]=1; hcost[src]=0;
    char hvis[1]; (void)hvis;
    Node *dn; int *de,*dcost,*dstop; char *dmask,*dupd,*dvis;
    cudaMalloc(&dn,sizeof(Node)*N); cudaMalloc(&de,sizeof(int)*E);
    cudaMalloc(&dcost,sizeof(int)*N); cudaMalloc(&dmask,N); cudaMalloc(&dupd,N);
    cudaMalloc(&dvis,N); cudaMalloc(&dstop,sizeof(int));
    cudaMemcpy(dn,hn,sizeof(Node)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(de,he,sizeof(int)*E,cudaMemcpyHostToDevice);
    cudaMemcpy(dcost,hcost,sizeof(int)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dmask,hmask,N,cudaMemcpyHostToDevice);
    cudaMemset(dupd,0,N);
    char *hvisinit=(char*)calloc(N,1); hvisinit[src]=1;
    cudaMemcpy(dvis,hvisinit,N,cudaMemcpyHostToDevice);
    int threads=512, blocks=(N+threads-1)/threads;
    int hstop;
    do {
        cudaMemset(dstop,0,sizeof(int));
        expand<<<blocks,threads>>>(dn,de,dmask,dupd,dvis,dcost,N);
        commit<<<blocks,threads>>>(dmask,dupd,dvis,N,dstop);
        cudaMemcpy(&hstop,dstop,sizeof(int),cudaMemcpyDeviceToHost);
    } while(hstop);
    cudaMemcpy(hcost,dcost,sizeof(int)*N,cudaMemcpyDeviceToHost);
    FILE *o=fopen(argv[2],"w");
    for(int i=0;i<N;i++) fprintf(o,"%d\t%d\n",i,hcost[i]);
    fclose(o);
    return 0;
}
