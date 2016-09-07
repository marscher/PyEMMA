
#ifndef __link_rmsd_unix_h
#define __link_rmsd_unix_h
#include <clustering.h>
#include <dlfcn.h>

void* load_minRMSD_lib() {
 void* handle;
 char* path = getenv("PYEMMA_CLUSTERING_LD");
 if (! path) {
    printf("set the correct path of PYEMMA_CLUSTERING_LD to point to the directory of the minRMSD_metric lib.\n");
    return NULL;
 }

 char* fn = "minRMSD_metric.so";
 char* abs_path = malloc((strlen(path) + strlen(fn) + 2)*sizeof(char));
 sprintf(abs_path, "%s/%s", path, fn);
 handle = dlopen(abs_path, RTLD_LAZY);
 free(abs_path);
 return handle;
}


distance_fptr load_minRMSD_distance() {
 distance_fptr p;
 char* err;
 void* minRMSD_metric;
 minRMSD_metric = load_minRMSD_lib();
 p = (distance_fptr) dlsym(minRMSD_metric, "minRMSD_distance");

 if ((err = dlerror()) != NULL) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", err);
 } else {
   return p;
 }

 return NULL;
}


int inplace_center_and_trace_atom_major_cluster_centers(float* centers_precentered, float* traces_centers_p,
    const int N_centers, const int dim) {
 char* err;
 typedef void (*center_fptr) (float*, float*, const int, const int);
 center_fptr p;
 void* minRMSD_metric;
 minRMSD_metric = load_minRMSD_lib();

 p = dlsym(minRMSD_metric, "inplace_center_and_trace_atom_major_cluster_centers_impl");
 if ((err = dlerror()) != NULL) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", err);
  return 1;
 } else {
  p(centers_precentered, traces_centers_p, N_centers, dim/3);
 }

 return 0;
}


#endif