#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define MAX_PD (3.0e6)
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
typedef float FLOAT;
const FLOAT t_chip=0.0005f, chip_height=0.016f, chip_width=0.016f, amb_temp=80.0f;
#define BLOCK_X 16
#define BLOCK_Y 16
#define PYRAMID 4
#define HALO (PYRAMID)
#define SM_X (BLOCK_X + 2 * HALO)
#define SM_Y (BLOCK_Y + 2 * HALO)
__device__ __forceinline__ FLOAT cell_update(FLOAT center,FLOAT up,FLOAT down,FLOAT left,FLOAT right,FLOAT pwr,int r,int c,int row,int col,FLOAT Cap_1,FLOAT Rx_1,FLOAT Ry_1,FLOAT Rz_1,FLOAT amb){
    FLOAT delta;
    if(r==0&&c==0){delta=(Cap_1)*(pwr+(right-center)*Rx_1+(down-center)*Ry_1+(amb-center)*Rz_1);}
    else if(r==0&&c==col-1){delta=(Cap_1)*(pwr+(left-center)*Rx_1+(down-center)*Ry_1+(amb-center)*Rz_1);}
    else if(r==row-1&&c==col-1){delta=(Cap_1)*(pwr+(left-center)*Rx_1+(up-center)*Ry_1+(amb-center)*Rz_1);}
    else if(r==row-1&&c==0){delta=(Cap_1)*(pwr+(right-center)*Rx_1+(up-center)*Ry_1+(amb-center)*Rz_1);}
    else if(r==0){delta=(Cap_1)*(pwr+(left+right-2.0f*center)*Rx_1+(down-center)*Ry_1+(amb-center)*Rz_1);}
    else if(c==col-1){delta=(Cap_1)*(pwr+(up+down-2.0f*center)*Ry_1+(left-center)*Rx_1+(amb-center)*Rz_1);}
    else if(r==row-1){delta=(Cap_1)*(pwr+(left+right-2.0f*center)*Rx_1+(up-center)*Ry_1+(amb-center)*Rz_1);}
    else if(c==0){delta=(Cap_1)*(pwr+(up+down-2.0f*center)*Ry_1+(right-center)*Rx_1+(amb-center)*Rz_1);}
    else{delta=(Cap_1)*(pwr+(left+right-2.0f*center)*Rx_1+(up+down-2.0f*center)*Ry_1+(amb-center)*Rz_1);}
    return center+delta;
}
__global__ void pyramid_kernel(FLOAT *dst,const FLOAT *src,const FLOAT *power,int row,int col,FLOAT Cap_1,FLOAT Rx_1,FLOAT Ry_1,FLOAT Rz_1,FLOAT amb,int steps_this_launch){
    __shared__ FLOAT s_t[SM_Y][SM_X]; __shared__ FLOAT s_n[SM_Y][SM_X]; __shared__ FLOAT s_p[SM_Y][SM_X];
    int base_r=blockIdx.y*BLOCK_Y; int base_c=blockIdx.x*BLOCK_X; int origin_r=base_r-HALO; int origin_c=base_c-HALO;
    for(int ly=threadIdx.y;ly<SM_Y;ly+=BLOCK_Y)for(int lx=threadIdx.x;lx<SM_X;lx+=BLOCK_X){
        int gr=origin_r+ly,gc=origin_c+lx; int cr=gr<0?0:(gr>=row?row-1:gr); int cc=gc<0?0:(gc>=col?col-1:gc);
        int gidx=cr*col+cc; s_t[ly][lx]=src[gidx]; s_p[ly][lx]=power[gidx];
    }
    __syncthreads();
    for(int k=0;k<steps_this_launch;++k){
        int margin=k+1;
        for(int ly=threadIdx.y;ly<SM_Y;ly+=BLOCK_Y)for(int lx=threadIdx.x;lx<SM_X;lx+=BLOCK_X){
            if(ly>=margin&&ly<SM_Y-margin&&lx>=margin&&lx<SM_X-margin){
                int gr=origin_r+ly,gc=origin_c+lx;
                if(gr>=0&&gr<row&&gc>=0&&gc<col){
                    FLOAT center=s_t[ly][lx],up=s_t[ly-1][lx],down=s_t[ly+1][lx],left=s_t[ly][lx-1],right=s_t[ly][lx+1];
                    s_n[ly][lx]=cell_update(center,up,down,left,right,s_p[ly][lx],gr,gc,row,col,Cap_1,Rx_1,Ry_1,Rz_1,amb);
                } else s_n[ly][lx]=s_t[ly][lx];
            } else s_n[ly][lx]=s_t[ly][lx];
        }
        __syncthreads();
        for(int ly=threadIdx.y;ly<SM_Y;ly+=BLOCK_Y)for(int lx=threadIdx.x;lx<SM_X;lx+=BLOCK_X) s_t[ly][lx]=s_n[ly][lx];
        __syncthreads();
    }
    int ly=threadIdx.y+HALO,lx=threadIdx.x+HALO,gr=base_r+threadIdx.y,gc=base_c+threadIdx.x;
    if(gr<row&&gc<col) dst[gr*col+gc]=s_t[ly][lx];
}

void fatal(const char *s){ fprintf(stderr,"%s\n",s); exit(1); }

void read_input(FLOAT *v,int grid_rows,int grid_cols,const char *file){
    FILE *fp=fopen(file,"r");
    if(!fp) fatal("file open error");
    int n=grid_rows*grid_cols;
    char str[256]; FLOAT val;
    for(int i=0;i<n;i++){
        if(fgets(str,256,fp)==NULL) fatal("Error reading file");
        if(feof(fp)) fatal("not enough lines in file");
        if(sscanf(str,"%f",&val)!=1) fatal("invalid file format");
        v[i]=val;
    }
    fclose(fp);
}

void write_output(FLOAT *v,int grid_rows,int grid_cols,const char *file){
    FILE *fp=fopen(file,"w");
    if(!fp) fatal("file open error");
    int n=grid_rows*grid_cols;
    char str[256];
    for(int i=0;i<n;i++){
        sprintf(str,"%d\t%g\n",i,v[i]);
        fputs(str,fp);
    }
    fclose(fp);
}

int main(int argc,char **argv){
    if(argc!=7){
        fprintf(stderr,"Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n",argv[0]);
        exit(1);
    }
    int grid_rows=atoi(argv[1]);
    int grid_cols=atoi(argv[2]);
    int sim_time=atoi(argv[3]);
    char *tfile=argv[4];
    char *pfile=argv[5];
    char *ofile=argv[6];
    if(grid_rows<=0||grid_cols<=0||sim_time<=0) fatal("invalid arguments");

    int n=grid_rows*grid_cols;
    FLOAT *temp=(FLOAT*)malloc(n*sizeof(FLOAT));
    FLOAT *power=(FLOAT*)malloc(n*sizeof(FLOAT));
    FLOAT *result=(FLOAT*)malloc(n*sizeof(FLOAT));
    if(!temp||!power||!result) fatal("malloc failed");

    read_input(temp,grid_rows,grid_cols,tfile);
    read_input(power,grid_rows,grid_cols,pfile);

    FLOAT grid_height=chip_height/grid_rows;
    FLOAT grid_width=chip_width/grid_cols;
    FLOAT Cap=FACTOR_CHIP*SPEC_HEAT_SI*t_chip*grid_width*grid_height;
    FLOAT Rx=grid_width/(2.0f*K_SI*t_chip*grid_height);
    FLOAT Ry=grid_height/(2.0f*K_SI*t_chip*grid_width);
    FLOAT Rz=t_chip/(K_SI*grid_height*grid_width);
    FLOAT max_slope=MAX_PD/(FACTOR_CHIP*t_chip*SPEC_HEAT_SI);
    FLOAT step=PRECISION/max_slope/1000.0f;   // FIX: extra /1000.0 to match serial reference
    FLOAT Cap_1=step/Cap;
    FLOAT Rx_1=1.0f/Rx;
    FLOAT Ry_1=1.0f/Ry;
    FLOAT Rz_1=1.0f/Rz;
    FLOAT amb=amb_temp;

    FLOAT *d_src,*d_dst,*d_power;
    cudaMalloc(&d_src,n*sizeof(FLOAT));
    cudaMalloc(&d_dst,n*sizeof(FLOAT));
    cudaMalloc(&d_power,n*sizeof(FLOAT));
    cudaMemcpy(d_src,temp,n*sizeof(FLOAT),cudaMemcpyHostToDevice);
    cudaMemcpy(d_power,power,n*sizeof(FLOAT),cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid((grid_cols+BLOCK_X-1)/BLOCK_X,(grid_rows+BLOCK_Y-1)/BLOCK_Y);

    int done=0;
    while(done<sim_time){
        int steps=PYRAMID;
        if(steps>sim_time-done) steps=sim_time-done;
        pyramid_kernel<<<grid,block>>>(d_dst,d_src,d_power,grid_rows,grid_cols,Cap_1,Rx_1,Ry_1,Rz_1,amb,steps);
        FLOAT *tmp=d_src; d_src=d_dst; d_dst=tmp;
        done+=steps;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(result,d_src,n*sizeof(FLOAT),cudaMemcpyDeviceToHost);
    write_output(result,grid_rows,grid_cols,ofile);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_power);
    free(temp); free(power); free(result);
    return 0;
}
