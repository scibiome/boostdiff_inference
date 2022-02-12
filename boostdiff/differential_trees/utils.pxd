import numpy as np
cimport numpy as np
# from tree_ver8 cimport Node

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from boostdiff.differential_trees.arandom cimport our_rand_r
from boostdiff.differential_trees.diff_tree cimport Node

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (StackRecord*)
    (Node*)
    (Node**)


cdef realloc_ptr safe_realloc(realloc_ptr* p, SIZE_t nelems) nogil except *


cdef np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)


cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil


cdef double rand_uniform(double low, double high,
                         UINT32_t* random_state) nogil


cdef double log(double x) nogil

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    
    SIZE_t start_d
    SIZE_t end_d
    SIZE_t start_c
    SIZE_t end_c
    SIZE_t depth
    SIZE_t parent
    bint is_left
    DTYPE_t diff_impurity
    DTYPE_t impurity_disease
    DTYPE_t impurity_control
    DTYPE_t original_impurity_disease
    DTYPE_t original_impurity_control
    DTYPE_t test
    DTYPE_t improvement_disease
    DTYPE_t improvement_control
    DTYPE_t differential_improvement
    DTYPE_t objective
        

cdef class Stack:
    
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    
    cdef int push(self, SIZE_t start_d, SIZE_t end_d, SIZE_t start_c, SIZE_t end_c,
                  SIZE_t depth, SIZE_t parent, bint is_left, DTYPE_t diff_impurity, 
                  DTYPE_t impurity_disease, DTYPE_t impurity_control, 
                  DTYPE_t original_impurity_disease, DTYPE_t original_impurity_control, 
                  DTYPE_t improvement_disease, DTYPE_t improvement_control, 
                  DTYPE_t differential_improvement,
                  DTYPE_t objective) nogil except -1
    
    cdef int pop(self, StackRecord* res) nogil
