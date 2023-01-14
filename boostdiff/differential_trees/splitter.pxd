import numpy as np
import math
cimport numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.math cimport floor, ceil, round
from libc.math cimport fabs


ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


# cdef inline double my_abs(double a) nogil:
    
#     return fabs(a)

from cpython cimport array
import array


cdef struct SplitRecord:
    
    SIZE_t feature
    double threshold_d 
    double threshold_c
    SIZE_t pos_d
    SIZE_t pos_c
    
    double impurity_improvement
    double original_impurity_disease
    double original_impurity_control
    double diff_impurity
    double objective
    double impurity_disease_left
    double impurity_disease_right
    double impurity_control_left
    double impurity_control_right
    double impurity_disease
    double impurity_control
    
    double improvement_disease
    double improvement_control
    double differential_improvement
    
    SIZE_t start_d
    SIZE_t end_d
    SIZE_t start_c
    SIZE_t end_c
    
# =============================================================================
# Splitter
# =============================================================================


cdef class Splitter:
        
    # Internal structures
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    
    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state      

    cdef const double[:, :] X_disease
    cdef const double[:, :] X_control
    
    cdef double[:] y_disease        # Values of y
    cdef double[:] y_control      # Values of y
    
    cdef SIZE_t* samples_disease                # Sample indices in X, y
    cdef SIZE_t* samples_control             # Sample indices in X, y
    
    cdef SIZE_t* features
    
    cdef double* feature_values_disease
    cdef double* feature_values_control
    
    cdef SIZE_t n_features
    
    cdef SIZE_t start_d
    cdef SIZE_t end_d
    cdef SIZE_t pos_d
    cdef SIZE_t start_c
    cdef SIZE_t end_c
    cdef SIZE_t pos_c
    
    cdef double sum_left_disease
    cdef double sum_right_disease
    cdef double sum_total_disease
    cdef double sq_sum_left_disease
    cdef double sq_sum_right_disease
    cdef double sq_sum_total_disease 
    
    cdef double sum_left_control
    cdef double sum_right_control
    cdef double sum_total_control
    cdef double sq_sum_left_control
    cdef double sq_sum_right_control
    cdef double sq_sum_total_control
    
    cdef SIZE_t n_node_samples_disease
    cdef SIZE_t n_node_samples_control
    
    cdef SIZE_t n_samples_disease
    cdef SIZE_t n_samples_control
    
    cdef SIZE_t n_left_d
    cdef SIZE_t n_right_d
    cdef SIZE_t n_left_c
    cdef SIZE_t n_right_c
    
    cdef int init(self, object X_disease, object X_control, double[:] y_disease, double[:] y_control) except -1
    
    cdef int node_reset(self, SIZE_t start_d, SIZE_t end_d, SIZE_t start_c, SIZE_t end_c) nogil except -1
    cdef void node_value(self, double* value) nogil
    
    cdef int reset_disease(self) nogil except -1
    cdef int reset_control(self) nogil except -1
    cdef int reverse_reset_disease(self) nogil except -1
    cdef int reverse_reset_control(self) nogil except -1
    
    cdef int update_disease(self, SIZE_t new_pos_d) nogil except -1
    cdef int update_control(self, SIZE_t new_pos_c) nogil except -1
    
    cdef double node_impurity_disease(self) nogil
    cdef double node_impurity_control(self) nogil
    
    cdef int children_impurity_improvement_disease_mse(self, SIZE_t start_d, SIZE_t pos_d, SIZE_t end_d, double impurity_parent_disease, double *improvement_disease, double *mse_disease, double *mse_disease_left, double *mse_disease_right) nogil except -1
    cdef int children_impurity_improvement_control_mse(self, SIZE_t start_c, SIZE_t pos_c, SIZE_t end_c, double impurity_parent_control, double *improvement_control, double *mse_control, double *mse_control_left, double *mse_control_right) nogil except -1
    
    cdef int find_thresholds_disease(self, SIZE_t current_feature, double impurity_parent_disease, SplitRecord* current_best) nogil except -1
    cdef int find_thresholds_control(self, SIZE_t current_feature, double impurity_parent_control, SplitRecord* current_best_control) nogil except -1
    
    cdef int node_split(self, SIZE_t start_d, SIZE_t end_d, SIZE_t start_c, SIZE_t end_c, double impurity_parent_disease, double impurity_parent_control, SplitRecord* split) nogil except -1