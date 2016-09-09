
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

 printf("trying to open: %s\n", path);
 handle = dlopen(path, RTLD_LAZY|RTLD_GLOBAL);
 return handle;
}


distance_fptr load_minRMSD_distance(void* module) {
 distance_fptr p;
 char* err;
 p = (distance_fptr) dlsym(module, "minRMSD_distance_impl");

 if ((err = dlerror()) != NULL) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", err);
 } else {
   return p;
 }

 return NULL;
}


center_fptr load_minRMSD_precenter(void* module) {
 char* err;
 center_fptr p;

 p = dlsym(module, "inplace_center_and_trace_atom_major_cluster_centers_impl");
 if ((err = dlerror()) != NULL) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", err);
  return NULL;
 }

 return p;
}


#endif