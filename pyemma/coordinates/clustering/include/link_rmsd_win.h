#ifndef __link_rmsd_win_h
#define __link_rmsd_win_h

#include <clustering.h>
#include <windows.h>

void* load_minRMSD_lib() {
 void* handle;
 char* path;
 path = getenv("PYEMMA_CLUSTERING_LD");
 if (! path) {
    printf("set the correct path of PYEMMA_CLUSTERING_LD to point to the directory of the minRMSD_metric lib.\n");
    return NULL;
 }

 handle = LoadLibrary(path);
 free(abs_path);
 return handle;
}


distance_fptr load_minRMSD_distance(void* module) {
 distance_fptr p;
 p = (distance_fptr) GetProcAddress(module, TEXT("minRMSD_distance_impl"));

 if (!p) { printf("win: error during loading %s\n", GetLastError()); }

 return p;
}

center_fptr load_minRMSD_precenter(void* module) {
 center_fptr p;

 p = (center_fptr) GetProcAddress(module, TEXT("inplace_center_and_trace_atom_major_cluster_centers_impl"));
 if (!p) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", GetLastError());
  return NULL;
 }
 return p;
}

#endif