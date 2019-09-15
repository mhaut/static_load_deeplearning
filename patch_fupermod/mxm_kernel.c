/*! Kernel of the \ref mxm routine */

#include "config.h"
#include "fupermod/fupermod_debug.h"
#include "fupermod/fupermod_kernel.h"
#include "mxm_2d.h"
#include "mxm_kernel.h"
#include "cblas_wrappers/fupermod_cblas.h"
#ifdef HAVE_LIBCUBLAS
#include "cblas_wrappers/cuda_auxiliary.hpp"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERBOSE_MATRIX 0

/*! Switch to choose using MPI_SEND/Recv to self or memcpy */
/* RICO */
//#define KERNEL_MPI_COPY 1
#define KERNEL_MPI_COPY 0

// Overall problem size
//TODO: N isn't used! Not relavent to kernel.
int N = 50;
char * argv[1];

// Block size
int kb = 32;
char buffer[256];
char conf_file[256];
char path_file[256];
// Number of iterations to benchmark
// RICO: ERROR: in execute_2d memcopys() iterate k on iters, so
//       we have an error because memory allocated is only for one iter.
//       So, we need iters = 1
//       How this affect to the accuracy???
int iters = 1;

// configuration for this process 
fupermod_process_conf* process_conf = NULL;

fupermod_float *A = NULL;
fupermod_float *B = NULL;
fupermod_float *C = NULL;

MPI_Comm comm = MPI_COMM_NULL;
fupermod_gemm* gemm = NULL;

int load(MPI_Comm _comm, fupermod_process_conf* conf) {
  if (conf) {
    char* subopts = strdup(conf->subopts);
    char* subopts_0 = subopts;
    char* tokens[] = {"N", "k", "i", "p", "P", NULL};
    char* value;

    while (*subopts != '\0') {
      switch (getsubopt(&subopts, tokens, &value)) {
        case 0:
          N = atoi(value);
          break;
        case 1:
          kb = atoi(value);
          break;
        case 2:
          iters = atoi(value);
        case 3:
          strcpy(path_file, value);
        case 4:
          strcpy(conf_file, value);

          break;
      }
    }
    free(subopts_0);
    process_conf = conf;
  }
  comm = _comm;
        
  return FUPERMOD_SUCCESS;
}

double complexity_2d(long long int m, long long int n, void* param){
  NOT_USED(param);
  return 2.0 * (m * kb) * (n * kb) * (iters * kb);
}

double complexity_1dfrom2d(long long int m, void* param){
  if (param == NULL){
    fprintf(stderr, "Error: This function cannot be called with param == NULL. Func:%s\n", __func__);
    MPI_Abort(comm, 30);
  }
  fupermod_model* model = (fupermod_model*)param;
  double* factor = (double*)model->params;
  return 2.0 * (m * kb) * (*factor * kb) * (iters * kb);
}


int myw, myh, reps;
fupermod_float* WA = NULL;
fupermod_float* WB = NULL;

int initialize_2d(long long int m, long long int n, void** params){
  NOT_USED(params);
  gemm = fupermod_gemm_alloc(process_conf);
  reps = 1;
  myh = m;
  myw = n;
  // initialise A, B, C, WA, WB
  A = (fupermod_float*) malloc(sizeof(fupermod_float) * myw * kb * myh * kb);
  B = (fupermod_float*) malloc(sizeof(fupermod_float) * myw * kb * myh * kb);
  C  = (fupermod_float*)fupermod_malloc(sizeof(fupermod_float) * myw * kb * myh * kb, process_conf);
  WA = (fupermod_float*)fupermod_malloc(sizeof(fupermod_float) * myh * kb * kb, process_conf);
  WB = (fupermod_float*)fupermod_malloc(sizeof(fupermod_float) * myw * kb * kb, process_conf);

  // Did it work?
  if (A == NULL || B == NULL || C == NULL || WA == NULL || WB == NULL) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "P%d Failed to allocate matrix\n"
        "P%d myw = %d, myh = %d, kb = %d\n"
        "p%d %s returning %d\n",rank, rank, myw, myh, kb, rank, __func__, FUPERMOD_ALLOC_FAIL );
    //*time = FUPERMOD_MAX_TIME;
    free(A);
    free(B);
    fupermod_free(C,  process_conf);
    fupermod_free(WA, process_conf);
    fupermod_free(WB, process_conf);
    fupermod_gemm_free(gemm);
    return FUPERMOD_ALLOC_FAIL;
  }
  // Fill A, B. Zero C
  int i;
  for (i = 0; i < myh * kb * myw * kb; i++) {
    A[i] = 1.1;
    B[i] = 2.5;
    C[i] = 0.0;
  }
  // Pull WA, WB into main memory, possibly push out A,B,C
  for (i = 0; i < myh * kb * kb; i++) {
    WA[i] = 0.0;
  }
  for (i = 0; i < myw * kb * kb; i++) {
    WB[i] = 0.0;
  }
  //
  if (process_conf && !strcmp(process_conf->device_type, "gpu")) {
    #ifdef HAVE_LIBCUBLAS
    cublasxgemm_params* params = (cublasxgemm_params*)gemm->params;
    params->workspace_WA = WA;
    params->workspace_WB = WB;
    #else
    fprintf(stderr, "%s : %s: Error no cuda library\n", __FILE__, __func__);
    #endif
  }  
  return FUPERMOD_SUCCESS;
}

int execute_2d(pthread_mutex_t* mutex, void* params) {    
  NOT_USED(params);
  NOT_USED(mutex);
  MPI_Status statusesR;
  MPI_Request requestsS, requestsR;
  int k;
  if(reps == 0){
  	return FUPERMOD_SUCCESS;
  }
  for (k = 0; k < 1; k++) {
    //make the benchmark as close to the real routine as possible:
    //MPI communications on the same process
    if (KERNEL_MPI_COPY) {
      MPI_Datatype fupermod_col;
      MPI_Type_vector(myh * kb, kb, myw * kb, FUPERMOD_MPI_FLOAT, &fupermod_col);
      MPI_Type_commit(&fupermod_col);

      int rc;
      rc = MPI_Isend(&A[k * kb], 1, fupermod_col, 0, 0xf1, MPI_COMM_SELF, &requestsS);
      if (rc != MPI_SUCCESS) {
        printf("Error Sending A\n");
        return FUPERMOD_FAIL;
      }
      MPI_Type_free(&fupermod_col);
      rc = MPI_Irecv(WA, myh * kb * kb, FUPERMOD_MPI_FLOAT, 0, 0xf1, MPI_COMM_SELF, &requestsR);
      if (rc != MPI_SUCCESS) {
        printf("Error Recving A\n");
        return FUPERMOD_FAIL;
      }
      MPI_Waitall(1, &requestsR, &statusesR);
    } else { // If you want to avoid mpi copy do this:
      int i;  
      for (i = 0; i < myh * kb ; i++) {
        memcpy(&WA[i * kb], &A[myw * kb * i +(k * kb)], sizeof(fupermod_float) * kb);
      }
    }
    fupermod_float* B_local = &B[(k * kb) * (myw * kb)];
    memcpy(WB, B_local, sizeof(fupermod_float) * myw * kb * kb);

    // multiply WA x WB = C
//    printf("myh: %d", myh*myw);

    int size = myh*myw;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//    char conf_file[256] = "m";
    printf("---------------Size:%d/t", world_size);
    printf(" ---------Reps:%d\t", reps);
    //char conf_file[256] = "/home/jarico/ws/t-lop/hetbatch/fpm/conf_file";

//    char conf_file[256] = "/mnt/shared/jarico/fupermod-latest/het_tests/Metropolis/SimpleModel_CIFAR10/P5/fpm/conf_file";
    snprintf(buffer, sizeof(buffer), "python %s %d %d %s %d", path_file, size, world_rank, conf_file, world_size);
    printf("\nWAITING %d\n", world_rank);
    system(buffer);
    printf("\nEXITING %d\n", world_rank);
    reps = reps - 1;

  }
  return FUPERMOD_SUCCESS;
}

int finalize_2d(void* params){
  printf("finalize start\n");
  NOT_USED(params);
  free(A);
  free(B);
  fupermod_free(C,  process_conf);
  fupermod_free(WA, process_conf);
  fupermod_free(WB, process_conf);
  //fupermod_gemm_free(gemm);
  printf("finalize end\n");
  return FUPERMOD_SUCCESS;
}
