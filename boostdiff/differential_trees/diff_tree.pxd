# -------------------------------------------------------------------------------
# This file is part of BoostDiff.
#
# BoostDiff incorporates work that is part of scikit-learn.
# See the original license notice below.
# -------------------------------------------------------------------------------

# BSD 3-Clause License

# Copyright (c) 2007-2021 The scikit-learn developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
cimport numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.math cimport floor, ceil, round

from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX


ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

from cpython cimport array
import array


cdef double INFINITY = np.inf

from boostdiff.differential_trees.splitter cimport Splitter, SplitRecord
    
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
    
    
cdef struct Node:
    
    # Base storage structure for the nodes in a Tree object
    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    SIZE_t depth                      
    double threshold_d                   # Threshold value at the node
    double threshold_c                   # Threshold value at the node
    double objective
    double diff_impurity                    # Impurity of the node (i.e., the value of the criterion)
    double impurity_disease
    double impurity_disease_left
    double impurity_disease_right
    double impurity_control
    double impurity_control_left
    double impurity_control_right
    double original_impurity_disease
    double original_impurity_control
    double improvement_disease
    double improvement_control
    double differential_improvement
    double test
    SIZE_t parent
    SIZE_t n_node_samples_dis                # Number of samples at the node
    SIZE_t n_node_samples_con               # Number of samples at the node
    

cdef class DiffTree:
    
    # Internal structures
    cdef SIZE_t min_samples_split      # Number of features to test
    cdef SIZE_t min_samples_leaf  # Min samples in a leaf
    
    cdef SIZE_t max_depth_seen
    cdef SIZE_t n_features
    
    cdef SIZE_t max_features
    
    cdef SIZE_t value_stride
    cdef SIZE_t n_outputs
    cdef SIZE_t max_n_classes
    
    cdef SIZE_t n_samples_disease
    cdef SIZE_t n_samples_control
    
    cdef SIZE_t max_depth
    
    cdef SIZE_t node_count
    cdef SIZE_t capacity
    
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    
    cdef Splitter splitter              # Splitting algorithm

    cdef SIZE_t add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                              SIZE_t feature, SIZE_t depth, double threshold_d, double threshold_c, 
                              double objective, double diff_impurity,
                              double original_impurity_disease, double original_impurity_control,
                              double improvement_disease, double improvement_control,
                              double differential_improvement, double test,
                              double impurity_disease, double impurity_disease_left, double impurity_disease_right,
                              double impurity_control, double impurity_control_left, double impurity_control_right,
                              SIZE_t n_node_samples_dis, SIZE_t n_node_samples_con) nogil except -1
    
    cpdef build(self, object X_disease, object X_control, np.ndarray y_disease, np.ndarray y_control, SIZE_t random_state)
    cdef inline np.ndarray _apply_dense(self, object X)
    cdef np.ndarray _get_value_ndarray(self)
    cpdef np.ndarray predict(self, object X)
    
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
        
    cpdef np.ndarray[np.float64_t, ndim=1] get_variable_importance_disease_gain(self)
    cpdef np.ndarray[np.float64_t, ndim=1] get_variable_importance_differential_improvement(self)
    
    cpdef np.ndarray[SIZE_t, ndim=1] get_selected_features(self)