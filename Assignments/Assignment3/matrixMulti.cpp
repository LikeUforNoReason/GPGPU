#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"

#define N 40
#define BLOCK_SIZE 1 

// OpenCL globals
cl_platform_id myplatform;
cl_context mycontext;
cl_device_id *mydevice;
cl_command_queue mycommandq;
cl_kernel mykernelfunc;
cl_program myprogram;
cl_mem gpuv_in1, gpuv_in2, gpuv_out;

cl_float *inputMatrix1;
cl_float *inputMatrix2;
cl_float *results;
cl_uint width = N;

void initCL()
{
int err;
size_t mycontxtsize, kernelsize;        // size_t is unsigned long (64 bits).
char *kernelsource;
unsigned int gpudevcount;

// Determine OpenCL platform
err = RGUGetPlatformID(&myplatform);
// Get number of GPU devices available on this platform:
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);

// Create and load the device list:
mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

for (int i=0; i<gpudevcount; i++) {
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        clGetDeviceInfo(mydevice[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(mydevice[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(mydevice[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(mydevice[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(mydevice[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
}

cl_context_properties props[] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)myplatform, 0};

// Create a compute context
mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
// Create a command queue
mycommandq = clCreateCommandQueue(mycontext,mydevice[0],CL_QUEUE_PROFILING_ENABLE,&err);

// Load kernel file, prepend static info, and return total kernel size.
kernelsource = RGULoadProgSource("myMatrixMulti.cl","", &kernelsize);
// arg0: file name of kernel to load
// arg1: preamble to prepend, e.g., .h file
// arg2: final length of code 

// Create program object and loads source strings into it.
myprogram = clCreateProgramWithSource(mycontext,1,
        (const char **)&kernelsource, NULL, NULL);
// arg1: number of string pointers in array of string pointers
// arg2: array of string pointers
// arg3: array of size_ts with lengths of strings; 
//       NULL==(all strings null-terminated)
// arg4: error code return

// Compile and link for all devices in context.
clBuildProgram(myprogram,0,NULL,NULL,NULL,NULL);
// arg1: number of devices in device list
// arg2: device list ptr; NULL == (use all devices in context)
// arg3: compile options; 
// arg4: callback function; called when compilation done; if NULL, suspend until
// arg5: data to callback function

if (clBuildProgram(myprogram,0,NULL,NULL,NULL,NULL) != CL_SUCCESS) {
        printf("Error building program.\n");

        char buffer[4096];
        size_t length;
        clGetProgramBuildInfo(myprogram, mydevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        printf("%s\n", buffer);
        exit(1);
}

// Create kernel object.
mykernelfunc = clCreateKernel(myprogram,"myMatrixMulti",NULL);
// arg1: kernel function name
// arg2: error code
}

void buffers()
{
// Create buffer objects == allocate mem on the card.
gpuv_in1 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        N*N*sizeof(float),inputMatrix1,NULL);
// RO means RO from a kernel; CL_MEM_COPY_HOST_PTR: alloc device memory 
// and copy data referenced by the host pointer; 
// arg4: error code return

gpuv_in2 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        N*N*sizeof(float),inputMatrix2,NULL);

gpuv_out = clCreateBuffer(mycontext,CL_MEM_WRITE_ONLY,N*N*sizeof(float),NULL,NULL);
// WO: written but not read from a kernel
}

void cleanup(int signo)
{
int i;
// Release GPU-allocated resources.
clReleaseProgram(myprogram);
clReleaseContext(mycontext);
clReleaseKernel(mykernelfunc);
clReleaseCommandQueue(mycommandq);
clReleaseMemObject(gpuv_in1);
clReleaseMemObject(gpuv_in2);
clReleaseMemObject(gpuv_out);
exit(0);
}

void zoom()
{
int i, j;
int N1 = 40;
// Set parameters to the kernel, "mykernelfunc".
clSetKernelArg(mykernelfunc,0,sizeof(int), &N1);
clSetKernelArg(mykernelfunc,1,sizeof(cl_mem),(void *)&gpuv_out);
clSetKernelArg(mykernelfunc,2,sizeof(cl_mem),(void *)&gpuv_in1);
clSetKernelArg(mykernelfunc,3,sizeof(cl_mem),(void *)&gpuv_in2);
clSetKernelArg(mykernelfunc,4,sizeof(cl_float)*width, NULL);
// arg1: which argument (L-to-R)
// arg2: size of argument; can use sizeof(type) for mem objects
// arg3: argument *

//global[0] = width;
//global[1] = width;
//local[0] = BLOCK_SIZE;
//local[1] = BLOCK_SIZE;

cl_uint status;
cl_event prof_event;
//cl_command_queue comm;

size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};
size_t global[2] = {width, width};

//mycommandq = clCreateCommandQueue(mycontext, mydevice[0], CL_QUEUE_PROFILING_ENABLE, &err);

// Launch the kernel.
status = clEnqueueNDRangeKernel(mycommandq,mykernelfunc,2,NULL,global,local,0,NULL,&prof_event);
// arg2: work dimension (of the grid)
// arg3: must be NULL; will be global work id offsets, instead of (0,0,...0)
// arg4: array of work dimension values giving number of work items in each
//       dim that will exec the kernel
// arg5: local work size - array of work dimension values giving work group
//       size in each dim; NULL = (OpenCL decides on work group sizes; 
//       Danger Will Robinson! OpenCL will make *BAD* decisions on this!)
// arg6: number of events in event waitlist
// arg7: event waitlist ... commands that must complete before exec this one
// arg8: event ... returns event object that identifies this kernel execution
//       instance; event objects are unique

clFinish(mycommandq);
status = clWaitForEvents(1, &prof_event);

//cl_ulong start_time, end_time;
cl_ulong start_time = (cl_ulong)0;
cl_ulong end_time = (cl_ulong)0;
size_t return_bytes;

status = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
status = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);

double run_time = (double)(end_time - start_time);
printf("Run time is : %.3f\n", run_time);
//cout << "Run time is : " << rum_time << endl;

// Read back the results.
clEnqueueReadBuffer(mycommandq,gpuv_out,CL_TRUE,0,N*N*sizeof(float),results,0,NULL,NULL);
// arg1: buffer object
// arg2: blocking read
// arg3: offset
// arg4: size in bytes
// arg5: host ptr
// arg6: number of events in event waitlist
// arg7: event waitlist ... commands that must complete before exec this one
// arg8: event ... returns event object that identifies this kernel execution
//       instance; event objects are unique
/*
printf("\nMatrix A:\n");
for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
             printf("%.3f\t", inputMatrix1[i*N + j]);
        printf("\n");
}

printf("\nMatrix B:\n");
for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
             printf("%.3f\t", inputMatrix2[i*N + j]);
        printf("\n");
}

printf("\nMatrix C:\n");
for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
             printf("%.3f\t", results[i*N + j]);
        printf("\n");
}
*/

}

int main(int argc, char** argv)
{
int x,y;
int data = 0;
inputMatrix1 = (cl_float *) malloc(sizeof(cl_float) * width * width);
inputMatrix2 = (cl_float *) malloc(sizeof(cl_float) * width * width);
results = (cl_float *) malloc(sizeof(cl_float) * width * width);

for (y = 0; y < width; y++) {
        for (x = 0; x < width; x++) {
                inputMatrix1[y * width + x] = data;
                inputMatrix2[y * width + x] = data;
                results[y * width + x] = 0;
                data++;
        }
}

signal(SIGUSR1,cleanup);
initCL();
buffers();
zoom();
cleanup(SIGUSR1);
}
