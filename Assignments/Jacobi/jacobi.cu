#include<stdio.h>
#include<stdlib.h>
#include<getopt.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>

static char* program_name;

// Usage
void print_usage (FILE* stream, int exit_code)
{
  fprintf (stream, "Usage:  %s options\n", program_name);
  fprintf (stream,
           "  -h  --help             Display this usage information.\n"
           "  -f  --file filename    File containing coefficient matrix.\n"
           "  -i  --Ny int           Number of elements in Y direction (default=512).\n"
           "  -j  --Nx int           Number of elements in X direction (default=512).\n"
           "  -n  --iterations int   Number of iterations (default=10000).\n"
           "  -k  --kernel [1,2]     1: unoptimized, 2: optimized kernel (default).\n"
           "  -t  --tilesize int     Size of each thread block in kernel 2 (default=4).\n");
  exit (exit_code);
}


// On the host
void jacobiHost(float* x1, float* A, float* x2, float* b, int Ny, int Nx)
{
    int i,j;
    float sigma;

    for (i=0; i<Ny; i++)
    {
        sigma = 0.0;
        for (j=0; j<Nx; j++)
        {
            if (i != j)
                sigma += A[i*Nx + j] * x2[j];
        }
        x1[i] = (b[i] - sigma) / A[i*Nx + i];
    }
}


// On the device 
__global__ void jacobiDevc(float* x1, float* A, float* x2, float* b, int Ny, int Nx)
{
    float sigma = 0.0;
    int idx = threadIdx.x;
    for (int j=0; j<Nx; j++)
    {
        if (idx != j)
            sigma += A[idx*Nx + j] * x2[j];
    }
    x1[idx] = (b[idx] - sigma) / A[idx*Nx + idx];
}


// Optimized the device
__global__ void jacobiOpDevc(float* x1, float* A, float* x2, float* b, int Ny, int Nx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
     
    if (idx < Ny)
    {
        float sigma = 0.0;
        int idx_Ai = idx*Nx;
        
        for (int j=0; j<Nx; j++)
            if (idx != j)
                sigma += A[idx_Ai + j] * x2[j];

        x1[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}


// Choose GPU
static void chooseGpu(int *gpuNum, int *devcNums)
{
    int m = *gpuNum;

    cudaGetDeviceCount(devcNums);
    if ( *devcNums > 1 )
    {
        int devcNum;
        int coresMax = 0;

        for (devcNum = 0; devcNum < *devcNums; devcNum++)
        {
            cudaDeviceProp devcProp;

            cudaGetDeviceProperties(&devcProp, devcNum);
            if (coresMax < devcProp.multiProcessorCount)
            {
                coresMax = devcProp.multiProcessorCount;
                m = devcNum;
            }
        }
        *gpuNum = m;
    }
}


// Test device
static void devcTest(int devcId)
{
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devcId);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
    {   
        printf("We can not find useful devices.\n");
        cudaThreadExit();
    }
    else
        printf("Using GPU device number %d.\n", devcId);
}


int main(int argc, char *argv[])
{ 
    time_t start, end, start_h, end_h, start_d, end_d;
    float t_full, t_host, t_dev;

    start=clock();

    float *x2, *x1, *A, *b, *x_h, *x_d;
    float *x2_d, *x1_d, *A_d, *b_d;
    int N, Ny, Nx, iter, kernel, tileSize;    
    int ch;
    int i,k;
    char* fname;
    FILE* file;

    static struct option long_options[] =
    {
        {"file", required_argument, NULL, 'f'},
        {"Ny", optional_argument, NULL, 'i'},
        {"Nx", optional_argument, NULL, 'j'},
        {"iterations", optional_argument, NULL, 'n'},
        {"kernel", optional_argument, NULL, 'k'},
        {"tilesize", optional_argument, NULL, 't'},
        {"help", optional_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    program_name = argv[0];
    Ny=512, Nx=512, iter=10000, kernel=2, tileSize=4;
    ch=0;
    
    while ((ch = getopt_long(argc, argv,"f:i:j:n:k:h", long_options, NULL)) != -1) {
        switch (ch) {
             case 'f' : fname = optarg;
                 break;
             case 'i' : Ny = atoi(optarg);
                 break;
             case 'j' : Nx = atoi(optarg); 
                 break;
             case 'n' : iter = atoi(optarg);
                 break;
             case 'k' : kernel = atoi(optarg);
                 break;
             case 't' : tileSize = atoi(optarg);
                 break;
             case 'h': print_usage(stderr, 1); 
                 exit(EXIT_FAILURE);
             case '?': print_usage(stderr, 1); 
                 exit(EXIT_FAILURE);
             default: 
                 abort();
        }
    }

    N = Ny * Nx;


    printf("\nRun Jacobi method:\n");
    printf("======================\n\n");
    printf("Coefficient matrix given in file: \n%s\n\n", fname);
    printf("Parameters:\n");
    printf("N=%d, Ny=%d, Nx=%d, ", N, Ny, Nx);
    printf("iterations=%d, kernel=%d, tilesize=%d\n", iter,kernel,tileSize);

    x1 = (float *) malloc(Ny*sizeof(float));
    A = (float *) malloc(N*sizeof(float));
    x2 = (float *) malloc(Ny*sizeof(float));
    b = (float *) malloc(Ny*sizeof(float));
    x_h = (float *) malloc(Ny*sizeof(float));
    x_d = (float *) malloc(Ny*sizeof(float));

    for (i=0; i<Ny; i++)
    {
        x2[i] = 0;
        x1[i] = 0;
    }

    file = fopen(fname, "r");
    if (file == NULL)
        exit(EXIT_FAILURE);
    char *line;
    size_t len = 0;
    i=0;
    while ((getline(&line, &len, file)) != -1) 
    {
        if (i<N)
            A[i] = atof(line);
        else
            b[i-N] = atof(line);
        i++;
    }
   

    start_h = clock();

    for (k=0; k<iter; k++)
    {
        if (k%2)
            jacobiHost(x2, A, x1, b, Ny, Nx);
        else
            jacobiHost(x1, A, x2, b, Ny, Nx);
    }
    
    end_h = clock();

    for (i=0; i<Nx; i++)
        x_h[i] = x1[i];


    for (i=0; i<Ny; i++)
    {
        x2[i] = 0;
        x1[i] = 0;
    }


    int devcId = 0, devcNums = 1;
    chooseGpu(&devcId, &devcNums);
    devcTest(devcId);
  
    assert(cudaSuccess == cudaMalloc((void **) &x1_d, Ny*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &A_d, N*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &x2_d, Ny*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &b_d, Ny*sizeof(float)));

    cudaMemcpy(x1_d, x1, sizeof(float)*Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(x2_d, x2, sizeof(float)*Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float)*Ny, cudaMemcpyHostToDevice);

    int blockSize = Ny;
    int nBlocks = 1;

    int nTiles = Ny/tileSize + (Ny%tileSize == 0?0:1);
    int gridHeight = Nx/tileSize + (Nx%tileSize == 0?0:1);
    int gridWidth = Ny/tileSize + (Ny%tileSize == 0?0:1);
    printf("w=%d, h=%d\n",gridWidth,gridHeight);
    dim3 dGrid(gridHeight, gridWidth),
        dBlock(tileSize, tileSize);


    start_d = clock();
     
    if (kernel == 1)
    {
        printf("Using the first kernel.\n");
        for (k=0; k<iter; k++)
        {
            if (k%2)
                jacobiDevc <<< nBlocks, blockSize >>> (x2_d, A_d, x1_d, b_d, Ny, Nx);
            else
                jacobiDevc <<< nBlocks, blockSize >>> (x1_d, A_d, x2_d, b_d, Ny, Nx);
        }
    }
    else
    {
        printf("Using the second kernel.\n");
        for (k=0; k<iter; k++)
        {
            if (k%2)
                jacobiOpDevc <<< nTiles, tileSize >>> (x2_d, A_d, x1_d, b_d, Ny, Nx);
            else
                jacobiOpDevc <<< nTiles, tileSize >>> (x1_d, A_d, x2_d, b_d, Ny, Nx);
        }
    }
        
    end_d = clock();

    cudaMemcpy(x_d, x1_d, sizeof(float)*Ny, cudaMemcpyDeviceToHost);
    
    free(x1); free(A); free(x2); free(b);
    cudaFree(x1_d); cudaFree(A_d); cudaFree(x2_d); cudaFree(b_d);

    end=clock(); 

    printf("\nResult after %d iterations:\n",iter);
    float err = 0.0;
    for (i=0; i < Ny; i++)
    {
        err += abs(x_h[i] - x_d[i]) / Ny;
    }
    printf("x_h[%d]=%f\n",0,x_h[0]);
    printf("x_d[%d]=%f\n",0,x_d[0]);
    t_full = ((float)end - (float)start) / CLOCKS_PER_SEC;
    t_host = ((float)end_h - (float)start_h) / CLOCKS_PER_SEC;
    t_dev = ((float)end_d - (float)start_d) / CLOCKS_PER_SEC;
    printf("\nTiming:\nFull: %f\nHost: %f\nDevice: %f\n\n", t_full, t_host, t_dev);
    printf("Relative error: %f\n", err);

    printf("\nProgram terminated successfully.\n");
    return 0;
}
