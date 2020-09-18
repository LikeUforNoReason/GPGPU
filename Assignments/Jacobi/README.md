Install:
    using the command './install.sh' to load the necessary CUDA module cuda/5.5.

Compile:
    execute the Makefile.

Run:
    excute the command './jacobi <arguments>'.
    The following command line arguments are available:
    -h  --help             Display this usage information.
    -f  --file filename    File containing coefficient matrix.
    -i  --Ny int           Number of elements in Y direction (default=512).
    -j  --Nx int           Number of elements in X direction (default=512).
    -n  --iterations int   Number of iterations (default=10000).
    -k  --kernel [1,2]     1: unoptimized, 2: optimized kernel (default).
    -t  --tilesize int     Size of each thread block in kernel 2 (default=4).

Input matrices of a particular size can be generated by 'python gen_diag_dominant_matrix.py <size> <output_filename>'. 