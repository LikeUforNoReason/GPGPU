// RGU.h - utilities
//
// These are my own utility functions to replace those from early versions
// of the NVIDIA SDK.  
//

cl_int RGUGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
char chBuffer[1024];
cl_uint num_platforms; 
cl_platform_id* clPlatformIDs;
cl_int err, i;
*clSelectedPlatformID = NULL;

// Get OpenCL platform count.
err = clGetPlatformIDs (0,NULL,&num_platforms);
if(err != CL_SUCCESS){
        fprintf(stderr,"eek clGetPlatformIDs\n");
        exit(1);
        }
if(num_platforms == 0){
        fprintf(stderr,"eek no platforms\n");
        exit(1);
        }
// If there's a platform or more, make space for ID's.
clPlatformIDs = (cl_platform_id*)calloc(num_platforms,sizeof(cl_platform_id));

// Get platform info for each platform. 
err = clGetPlatformIDs(num_platforms,clPlatformIDs,NULL);
for(i=0;i<num_platforms;i++) {
        err = clGetPlatformInfo(clPlatformIDs[i],CL_PLATFORM_NAME,1024,
                &chBuffer, NULL);
        if(err == CL_SUCCESS) {
                if(strstr(chBuffer,"NVIDIA")!=NULL){
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                        }
                }
        }
if(*clSelectedPlatformID==NULL) fprintf(stderr,"NVIDIA has disappeared.\n");
free(clPlatformIDs);
return CL_SUCCESS;
}

char* RGULoadProgSource(const char* filename,const char* preamble,size_t* sz)
{
FILE* fptr = NULL;
size_t szSource, szPreamble, howmany;
char* SourceString;

// Open the OpenCL source code file.
fptr = fopen(filename, "r");
szPreamble = strlen(preamble);

// Get the length of the source code.
fseek(fptr,0,SEEK_END);
szSource = ftell(fptr);
fseek(fptr,0,SEEK_SET);

// Allocate a buffer for the source code string and read it in.
SourceString = (char *)calloc(szSource+szPreamble+1,sizeof(char));
memcpy(SourceString,preamble,szPreamble);
howmany = fread((SourceString)+szPreamble,szSource,1,fptr);
fclose(fptr);
*sz = szSource + szPreamble;
SourceString[szSource+szPreamble] = '\0';
return SourceString;
}
                                                                                              68,1          Bot
