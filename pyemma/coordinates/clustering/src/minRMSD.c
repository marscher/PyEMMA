/**
 * minRMSD function wrappers around mdtraj, which should be built as a dynamic library
 * to avoid the mdtraj dependency during build time.
 */

#include <clustering.h>
#include <theobald_rmsd.h>
#include <center.h>

/*
 * minRMSD distance function
 * a: centers
 * b: frames
 * n: dimension of one frame
 * buffer_a: pre-allocated buffer to store a copy of centers
 * buffer_b: pre-allocated buffer to store a copy of frames
 * trace_a_precalc: pre-calculated trace to centers (pointer to one value)
 */
#ifdef __cplusplus
extern "C" {
#endif

float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n,
                       float *SKP_restrict buffer_a, float *SKP_restrict buffer_b,
                       float *trace_a_precalc)
{
    float msd;
    float trace_a, trace_b;

    if (! trace_a_precalc) {
        memcpy(buffer_a, a, n*sizeof(float));
        memcpy(buffer_b, b, n*sizeof(float));

        inplace_center_and_trace_atom_major(buffer_a, &trace_a, 1, n/3);
        inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);

    } else {
        // only copy b, since a has been pre-centered,
        memcpy(buffer_b, b, n*sizeof(float));
        inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);
        trace_a = *trace_a_precalc;
    }

    msd = msd_atom_major(n/3, n/3, a, buffer_b, trace_a, trace_b, 0, NULL);
    return sqrt(msd);
}

void inplace_center_and_trace_atom_major_cluster_centers_impl(float* centers_precentered, float* traces_centers_p,
    const int N_centers, const int dim) {
    int j;
    for (j = 0; j < N_centers; ++j) {
        inplace_center_and_trace_atom_major(&centers_precentered[j*dim],
                                            &traces_centers_p[j], 1, dim/3);
    }
}

// fake a module
static PyMethodDef kmeansMethods[] =
{
     {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}


#if PY_MAJOR_VERSION >= 3

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "minRMSD_metric",
        NULL,
        sizeof(struct module_state),
        kmeansMethods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_minRMSD_metric(void)

#else // py2
#define INITERROR return

void initminRMSD_metric(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3("minRMSD_metric", kmeansMethods, "");
#endif
    struct module_state *st = GETSTATE(module);

    if (module == NULL)
        INITERROR;

    st->error = PyErr_NewException("minRMSD_metric.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

#ifdef __cplusplus
}
#endif