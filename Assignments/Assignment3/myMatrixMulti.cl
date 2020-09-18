//myMatrixMulti.cl

__kernel void myMatrixMulti(const int N,__global float *c,__global float *a,__global float *b, __local float *Bwrk)
{
int j, k;
int i = get_global_id(0);
int iloc = get_local_id(0);
int nloc = get_local_size(0);
float tmp = 0.0f;
float Awrk[1024];
for (k = 0; k < N; ++k) {
	Awrk[k] = a[i*N+k];
}
for (j = 0; j < N; ++j) {
	for (k = iloc; k < N; k+=nloc)
		Bwrk[k] = b[k*N+j];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (k = 0; k < N; k++) 
		tmp += Awrk[k]*Bwrk[k];

	c[i*N+j] = tmp;

	barrier(CLK_LOCAL_MEM_FENCE);
}
}
