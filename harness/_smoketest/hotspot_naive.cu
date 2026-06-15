/* Harness smoke-test ONLY — minimal naive CUDA hotspot to validate the
 * evaluate.py plumbing (build/run/verify/timing). NOT an experiment artifact. */
#include <stdio.h>
#include <stdlib.h>

#define STR_SIZE 256
#define MAX_PD       (3.0e6)
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100
#define FACTOR_CHIP  0.5
typedef float FLOAT;
const FLOAT t_chip = 0.0005f, chip_height = 0.016f, chip_width = 0.016f, amb_temp = 80.0f;

__global__ void step(FLOAT *res, const FLOAT *temp, const FLOAT *power, int row, int col,
                     FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= row || c >= col) return;
    int i = r * col + c;
    FLOAT tc = temp[i];
    FLOAT left  = (c == 0)       ? tc : temp[i-1];
    FLOAT right = (c == col-1)   ? tc : temp[i+1];
    FLOAT up    = (r == 0)       ? tc : temp[i-col];
    FLOAT down  = (r == row-1)   ? tc : temp[i+col];
    FLOAT delta = Cap_1 * (power[i]
        + (right + left - 2.f*tc) * Rx_1
        + (up + down - 2.f*tc) * Ry_1
        + (amb_temp - tc) * Rz_1);
    res[i] = tc + delta;
}

void fatal(const char *s){ fprintf(stderr,"error: %s\n",s); exit(1);}
void read_input(FLOAT*v,int rr,int cc,const char*f){FILE*fp=fopen(f,"r");if(!fp)fatal("open");char s[STR_SIZE];FLOAT val;for(int i=0;i<rr*cc;i++){if(!fgets(s,STR_SIZE,fp))fatal("eof");if(sscanf(s,"%f",&val)!=1)fatal("fmt");v[i]=val;}fclose(fp);}
void writeoutput(FLOAT*v,int rr,int cc,const char*f){FILE*fp=fopen(f,"w");char s[STR_SIZE];int idx=0;for(int i=0;i<rr;i++)for(int j=0;j<cc;j++){sprintf(s,"%d\t%g\n",idx,v[i*cc+j]);fputs(s,fp);idx++;}fclose(fp);}

int main(int argc, char** argv){
    if(argc!=7){fprintf(stderr,"usage\n");return 1;}
    int row=atoi(argv[1]), col=atoi(argv[2]), sim=atoi(argv[3]);
    int N=row*col; size_t sz=N*sizeof(FLOAT);
    FLOAT *temp=(FLOAT*)malloc(sz), *power=(FLOAT*)malloc(sz), *result=(FLOAT*)malloc(sz);
    read_input(temp,row,col,argv[4]); read_input(power,row,col,argv[5]);

    FLOAT grid_height=chip_height/row, grid_width=chip_width/col;
    FLOAT Cap=FACTOR_CHIP*SPEC_HEAT_SI*t_chip*grid_width*grid_height;
    FLOAT Rx=grid_width/(2.0f*K_SI*t_chip*grid_height);
    FLOAT Ry=grid_height/(2.0f*K_SI*t_chip*grid_width);
    FLOAT Rz=t_chip/(K_SI*grid_height*grid_width);
    FLOAT max_slope=MAX_PD/(FACTOR_CHIP*t_chip*SPEC_HEAT_SI);
    FLOAT step_=PRECISION/max_slope/1000.0f;
    FLOAT Rx_1=1.f/Rx, Ry_1=1.f/Ry, Rz_1=1.f/Rz, Cap_1=step_/Cap;

    FLOAT *dT,*dR,*dP; cudaMalloc(&dT,sz);cudaMalloc(&dR,sz);cudaMalloc(&dP,sz);
    cudaMemcpy(dT,temp,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(dP,power,sz,cudaMemcpyHostToDevice);
    dim3 b(16,16), g((col+15)/16,(row+15)/16);
    for(int i=0;i<sim;i++){ step<<<g,b>>>(dR,dT,dP,row,col,Cap_1,Rx_1,Ry_1,Rz_1); FLOAT*t=dT;dT=dR;dR=t; }
    cudaDeviceSynchronize();
    cudaMemcpy(temp,dT,sz,cudaMemcpyDeviceToHost); /* latest is in dT after swaps */
    writeoutput(temp,row,col,argv[6]);
    cudaFree(dT);cudaFree(dR);cudaFree(dP); free(temp);free(power);free(result);
    return 0;
}
