#ifndef __link_rmsd_win_h
#define __link_rmsd_win_h

#include <clustering.h>
#include <windows.h>

typedef int (__cdecl *MYPROC)(LPWSTR);

void* load_minRMSD_lib() {
 void* handle;
 char* path;
 char* fn;
 char* abs_path;
 path = getenv("PYEMMA_CLUSTERING_LD");
 if (! path) {
    printf("set the correct path of PYEMMA_CLUSTERING_LD to point to the directory of the minRMSD_metric lib.\n");
    return NULL;
 }

 fn = "minRMSD_metric.pyd";
 abs_path = malloc((strlen(path) + strlen(fn) + 2)*sizeof(char));
 sprintf(abs_path, "%s/%s", path, fn);
 handle = LoadLibrary(abs_path);
 free(abs_path);
 return handle;
}


distance_fptr load_minRMSD_distance() {
 distance_fptr p;
 char* err;
 void* minRMSD_metric;
 minRMSD_metric = load_minRMSD_lib();
 if (!minRMSD_metric) { printf("win: error during loading %s\n", GetLastError()); }
 p = (distance_fptr) GetProcAddress(minRMSD_metric, TEXT("minRMSD_distance"));

 if (!p) { printf("win: error during loading %s\n", GetLastError()); }

 return p;
}

int inplace_center_and_trace_atom_major_cluster_centers(float* centers_precentered, float* traces_centers_p,
    const int N_centers, const int dim) {
 char* err;
 typedef void (__cdecl *center_fptr) (float*, float*, const int, const int);
 center_fptr p;
 void* minRMSD_metric;
 minRMSD_metric = load_minRMSD_lib();
 if (! minRMSD_metric) {
    return 1;
 }

 p = (center_fptr) GetProcAddress(minRMSD_metric, TEXT("inplace_center_and_trace_atom_major_cluster_centers_impl"));
 if (!p) {
  /* handle error, the symbol wasn't found */
  printf("error during loading: %s\n", GetLastError());
  return 1;
 } else {
  /* symbol found, its value is in s */
  p(centers_precentered, traces_centers_p, N_centers, dim/3);
 }
 return 0;
}

#endif