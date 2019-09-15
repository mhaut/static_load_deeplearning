/*!
 * \page builder Model builder
 * Builds full functional performance models over a range of problem sizes. 
 * May be run as a single process or in parallel. One model is output per MPI process.
 * Uses \ref fupermod_conf_group to set process specific parameters.
 * If conf_file exists, it is used, otherwise a default machine file is created for the current set of MPI processes. This can be further modified by the user and the builder run again.
 *
 * run $ ./builder -h for details of input parameters.
 *
 * Requires a routine kernel to be given as input to benchmark.
 *
 * Outline of programme:
 * \code
 *   - Read config from conf_file (create if necessary)
 *	 - open output file
 *	 - Print header info from options & subopts
 *	 - Loop over different problem sizes, using different methods.
 *	 - - Initialise kernel
 *	 - - loop over repetitions
 *	 - - - start time
 *	 - - - execute kernel
 *	 - - - end time
 *	 - - Finialise kernel
 *	 - - Statistics on times, mean, student-t
 *	 - - Print out times to file
 *	 - End Loop
 *	 - close file
 * \endcode
 */

#include "config.h"
#include "fupermod/fupermod_conf.h"
#include "fupermod/fupermod_data.h"
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char** argv) {

  char path[256]      = "";
  char p_fpm[256] = "";
  char conf_fpm[256] = "";
  char subopts[256]   = "";
  char data_dir[256] = "./";
  char* conf_file = "./conf_file";
  int debug     = 0;
  int exit      = 0;
  int max_steps = 3;
  int n_upper   = 100;
  int n_lower   = 1;
  int increment = 1;
  int method    = 0;
  double granularity = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  fupermod_precision precision = fupermod_precision_defaults();

  int rank;
  MPI_Comm_rank(comm, &rank);
  int root = 0;
  //if (rank == root) {
  int opt;
  while ((opt = getopt(argc, argv, "hl:f:o:m:L:U:g:c:s:r:R:T:i:e:z:Z:d:p:P")) != -1)
    switch (opt) {
      case 'h':
        if (rank == root) {
          fprintf(
              stderr,
              "Builds full functional performance model\n"
              "Usage: builder [options]\n"
              "	-h	help\n"
              "	-l S	shared library (required: path to LD_LIBRARY_PATH)\n"
              "	-f S	path to data file (default: %s)\n"
              "	-o S	suboptions for the shared library separated by comma\n"
              "	-m I	method (default %d)\n"
              "	     0 Fixed number of points set [L, U, s] \n"
              "	     1 Adaptive, set [L, U, g, c]\n"
              "	-L I	lower problem size (default:%d)\n"
              "	-U I	upper problem size (default:%d)\n"
              "	-g D	granularity of measurement (default %f)\n"
              "	-c I	initial increment (default %d)\n"
              "	-s I	number of steps in the model (default %d)\n"
              "	-r I	minimum number of repetitions to average over (>=1) (default %d)\n"
              "	-R I	maximum number of repititions (defaule %d)\n"
              "	-T D	Maximum time benchmarking a point (default %f)\n"
              "	-i D	confidence level (default %f)\n"
              "	-e D	relative error (default %f)\n"
              "	-z D	zero speed, below which benchmark is killed with pthreads, disabled with 0 (default %le).\n"
              "	-Z D	max time, which benchmark is killed with pthreads (default %le).\n"
              " -p S  path to neural network benchmark\n"
              " -P S  path to conf file\n"

              "	-d  	debug\n",
            data_dir, method, n_lower, n_upper, granularity, increment, max_steps,
            precision.reps_min, precision.reps_max, precision.time_max_rep,
            precision.cl, precision.eps, precision.zero_speed, precision.time_max_kill);
        }
        exit = 1;
        break;
      case 'l':
        strcpy(path, optarg);
        break;
      case 'f':
        strcpy(data_dir, optarg);
        break;
      case 'o':
        strcpy(subopts, optarg);
        break;
      case 'm':
        method = atoi(optarg);
        break;
      case 'L':
        n_lower = atoi(optarg);
        break;
      case 'U':
        n_upper = atoi(optarg);
        break;
      case 'g':
        granularity = atoi(optarg);
        break;
      case 'c':
        increment = atoi(optarg);
        break;
      case 's':
        max_steps = atoi(optarg);
        break;
      case 'r':
        precision.reps_min = atoi(optarg);
        break;
      case 'R':
        precision.reps_max = atoi(optarg);
        break;
      case 'T':
        precision.time_max_rep = atof(optarg);
        break;
      case 'z':
        precision.zero_speed = atof(optarg);
        break;
      case 'Z':
        precision.time_max_kill = atof(optarg);
        break;
      case 'i':
        precision.cl = atof(optarg);
        break;
      case 'e':
        precision.eps = atof(optarg);
        break;
      case 'd':
        debug = 1;
        break;
      case 'p':
        strcpy(p_fpm, optarg);
        break;
      case 'P':
        strcpy(conf_fpm, optarg); 
        break;

      default:
        fprintf(stderr, "Unknown option %s\n", optarg);
        break;
    }
  if (!strcmp(path, "")) {
    fprintf(stderr, "Error: routine shared library not specified\n");
    exit = 1;
  }
  //}
  MPI_Bcast(&exit, 1, MPI_INT, root, comm);
  if (exit) {
    MPI_Finalize();
    return 0;
  }
  MPI_Bcast(&data_dir, 256, MPI_CHAR, root, comm);
  MPI_Bcast(&method, 1, MPI_INT, root, comm);
  MPI_Bcast(&granularity, 1, MPI_DOUBLE, root, comm);
  MPI_Bcast(&increment, 1, MPI_INT, root, comm);
  MPI_Bcast(&max_steps, 1, MPI_INT, root, comm);
  MPI_Bcast(&debug, 1, MPI_INT, root, comm);
  MPI_Bcast(&precision, sizeof(fupermod_precision), MPI_CHAR, root, comm);

  if (debug) {
    if (!rank) {
      printf("pid %d\n", getpid());
      getc(stdin);
    }	
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /*TODO this won't works on a platform without nfs*/
  int file_exist = 0;
  if(rank == 0 && access(conf_file, R_OK) !=-1){
    file_exist = 1;
  }
  MPI_Bcast(&file_exist, 1, MPI_INT, root, comm);
  if(file_exist==0){
    FILE* file = fopen(conf_file, "w");
    MPI_Barrier(MPI_COMM_WORLD);
    if(file == NULL) {
      perror("can't create conf_file");
      MPI_Finalize();
      return 0;
    }
    fupermod_print_conf(MPI_COMM_WORLD, root, file, "cpu", subopts);
    fclose(file);
    fflush(file);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  // configuration
  fupermod_process_conf conf = fupermod_get_conf(MPI_COMM_WORLD, conf_file);
  // load library
  fupermod_library library;
  if(fupermod_library_load(comm, path, &conf, &library) != 0) {
    fprintf(stderr,"Error: opening shared library\n");
    return 1;
  }
  printf("[%d] Kernel:%s  %s (OMP #thrs: %d)\n", rank, library.path, library.conf->subopts, omp_get_num_procs());

  // data file  
  char* filename = fupermod_data_filename(data_dir, conf);
  FILE* outfile = fopen(filename, "w");
  free(filename); filename = NULL;
  if (outfile == NULL)
    MPI_Abort(comm, -1);
  fupermod_data_write_head(outfile, &library);
    
  // benchmark
  fupermod_benchmark* benchmark = fupermod_benchmark_basic_alloc(library.kernel, conf);
  MPI_Comm comm_intra;
  fupermod_comm_intra(MPI_COMM_WORLD, &comm_intra);
  
  if (max_steps < 2 || n_upper <= n_lower)
    increment = n_upper;
  else if (method == 0)
    increment = (int) (n_upper - n_lower) / (max_steps - 1);
  if (increment < 1)
    increment = 1;

    /* RICO */
    //increment = 8;
    
    //fprintf(stdout, "[%d] Starting computation with %d threads with increment %d (%d to %d)\n ", rank, omp_get_num_procs(), increment, n_lower, n_upper);

  int n;
  for (n  = n_lower; n <= n_upper; n += increment) {
    // Data point for a process
    if (n > 0) {
      fupermod_point point;
      benchmark->execute(benchmark, comm_intra, n, precision, &point);
      fupermod_point_write(outfile, &point, library.kernel->complexity);

      if (method == 1 && point.t / increment > granularity)
        increment *= 2;
    }
  }
  fclose(outfile);
  
  MPI_Comm_free(&comm_intra);
  fupermod_benchmark_basic_free(benchmark);
  fupermod_library_unload(comm, &library);
  fupermod_conf_free(conf);
  MPI_Finalize();
  return 0;
}
