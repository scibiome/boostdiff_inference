# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import math
import numpy as np
import pandas as pd
import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
np.import_array()

from boostdiff.differential_trees.utils cimport safe_realloc
from boostdiff.differential_trees.utils cimport Stack
from boostdiff.differential_trees.utils cimport StackRecord
from boostdiff.differential_trees.splitter cimport Splitter, SplitRecord

from cpython cimport Py_INCREF, PyObject, PyTypeObject

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 20
cdef double INFINITY = np.inf        

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)


cdef class DiffTree:
    
    def __cinit__(self, SIZE_t min_samples_leaf, SIZE_t min_samples_split, 
                  SIZE_t n_samples_disease, SIZE_t n_samples_control,
                  SIZE_t max_depth, SIZE_t max_features):
        
        # Initialize the various criteria to be set for splitting
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf 
        self.max_features = max_features
        
        self.n_features = 0
        
        # Inner structures
        self.max_depth_seen = -1
        self.max_depth = max_depth
        self.node_count = 0
        self.value_stride = 1
        
        self.n_samples_disease = n_samples_disease
        self.n_samples_control = n_samples_control
        
        self.value = NULL
        self.nodes = NULL
        
        self.capacity = 0
        
        self.n_outputs = 1
        self.max_n_classes = 1
        
        # self.splitter = splitter
        
        cdef Node dummy;
        NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype


    def __dealloc__(self):
        
        """Destructor."""
        
        # Free all inner structures
        free(self.value)
        free(self.nodes)
        
    cpdef np.ndarray[np.float64_t, ndim=1] get_variable_importance_disease_gain(self):
        
        """Computes the importance of each feature (aka variable)."""
        
        cdef Node left, right
        cdef Node node
        cdef SIZE_t i_node = 0
        cdef DOUBLE_t node_imp_temp
        cdef double normalizer = 0.
        cdef DOUBLE_t fraction
        
        cdef np.ndarray[np.float64_t, ndim=1] importances_disease
        importances_disease = np.zeros((self.n_features,))
        # print()
        # print("variable importance")
        # print("self.node_count", self.node_count)
        while i_node != self.node_count:
            
            node = self.nodes[i_node]
            # print("i_node", i_node)
            # print("node", node.feature, node.differential_improvement)
            # print()
            # print("node", node)

            if node.feature != _TREE_UNDEFINED:
                
                left = self.nodes[node.left_child]
                right = self.nodes[node.right_child]
                
                fraction = <DOUBLE_t>(<DOUBLE_t>node.n_node_samples_dis / <DOUBLE_t>self.n_samples_disease)
                node_imp_temp = <DOUBLE_t>(node.original_impurity_disease -<DOUBLE_t>(left.original_impurity_disease*(left.n_node_samples_dis/node.n_node_samples_dis)) - <DOUBLE_t>(right.original_impurity_disease*(right.n_node_samples_dis/node.n_node_samples_dis)))
                # with gil:
                # print("node.original_impurity_disease", node.original_impurity_disease)
                # print("node_imp_temp", node_imp_temp)
                # print("importances_disease[node.feature]", importances_disease[node.feature])
                importances_disease[node.feature] += <DOUBLE_t>(fraction*node_imp_temp)

            i_node += 1
           
        # print("importances_disease", importances_disease)
        return importances_disease
            
    
    cpdef np.ndarray[np.float64_t, ndim=1] get_variable_importance_differential_improvement(self):
        
        """Computes the importance of each feature (aka variable)."""
        
        cdef Node left, right
        cdef Node node
        cdef SIZE_t i_node = 0
        cdef DTYPE_t node_imp_temp
        
        cdef np.ndarray[DOUBLE_t, ndim=1] importances
        importances = np.zeros((self.n_features,))

        while i_node != self.node_count:
            
            node = self.nodes[i_node]
            
            if node.feature != TREE_UNDEFINED:
                
                # importances[node.feature] += <DOUBLE_t>fabs(node.differential_improvement)
                importances[node.feature] += <DOUBLE_t>node.differential_improvement
            i_node += 1
            
        return importances


    cpdef np.ndarray[SIZE_t, ndim=1] get_selected_features(self):
        
        """Computes the importance of each feature (aka variable)."""
        
        cdef Node left, right
        cdef Node node
        cdef SIZE_t i_node = 0
        
        cdef SIZE_t n_internals = <SIZE_t>(self.node_count - 1)/2
        # print("n_internals", n_internals)
        cdef SIZE_t split_counter = 0
        
        cdef np.ndarray[SIZE_t, ndim=1] selected_features
        selected_features = np.zeros((n_internals,), dtype=np.intp)

        while i_node != self.node_count:
            
            node = self.nodes[i_node]
            
            if node.feature != TREE_UNDEFINED:
                
                # importances[node.feature] += <DOUBLE_t>fabs(node.differential_improvement)
                selected_features[split_counter] = node.feature
                split_counter += 1
            i_node += 1
            
        return selected_features
    

    cdef SIZE_t add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, SIZE_t depth, double threshold_d, double threshold_c, 
                          double objective, double diff_impurity,
                          double original_impurity_disease, double original_impurity_control,
                          double improvement_disease, double improvement_control,
                          double differential_improvement, double test,
                          double impurity_disease, double impurity_disease_left, double impurity_disease_right,
                          double impurity_control, double impurity_control_left, double impurity_control_right,
                          SIZE_t n_node_samples_dis, SIZE_t n_node_samples_con) nogil except -1:

        """
        Add a node to the tree.
        The new node registers itself as the child of its parent.
        """

            
        # Counter for the no. of internal nodes
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX
        
        cdef int rc = 0
        cdef bint is_first = 1
        
        cdef Node* node = &self.nodes[node_id]
        
        node.objective = objective
        node.diff_impurity = diff_impurity
        node.impurity_disease = impurity_disease
        node.impurity_control = impurity_control
        node.impurity_disease_left = impurity_disease_left
        node.impurity_disease_right = impurity_disease_right
        node.impurity_control_left = impurity_control_left
        node.impurity_control_right = impurity_control_right
        node.original_impurity_disease = original_impurity_disease
        node.original_impurity_control = original_impurity_control
        node.improvement_disease = improvement_disease
        node.improvement_control = improvement_control
        node.differential_improvement = differential_improvement
        node.parent = parent
        node.feature = feature
        node.depth = depth
        node.n_node_samples_dis = n_node_samples_dis
        node.n_node_samples_con = n_node_samples_con
        node.test = differential_improvement

            
        if parent != _TREE_UNDEFINED:
            
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id


        if is_leaf:
            
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold_d = _TREE_UNDEFINED
            node.threshold_c = _TREE_UNDEFINED
            
        else:
            # Left_child and right_child will be set later
            node.feature = feature
            node.threshold_d = threshold_d
            node.threshold_c = threshold_c

        self.node_count += 1
        
        return node_id
    
    
    
    cpdef build(self, object X_disease, object X_control, np.ndarray y_disease, np.ndarray y_control, SIZE_t random_state):
        
        """Build a decision tree from the training set (X, y)."""

        # Initial capacity
        cdef int init_capacity

        if self.max_depth <= 10:
            init_capacity = (2 ** (self.max_depth + 1)) - 1
        else:
            init_capacity = 2047
       
        
        self._resize(init_capacity)
        
        # Set the number of features
        self.n_features = <SIZE_t>X_disease.shape[1]
        
        # Parameters
        # print("======TREE=======")
        # print("random_state", random_state)
        cdef Splitter splitter = Splitter(random_state, self.max_features)
        # self.splitter = splitter
        # cdef Splitter splitter = self.splitter
        
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        # Recursive partition (without actual recursion)
        splitter.init(X_disease, X_control, y_disease, y_control)

        cdef SIZE_t start_d
        cdef SIZE_t end_d
        cdef SIZE_t start_c
        cdef SIZE_t end_c
        
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        
        cdef SIZE_t n_node_samples_disease = splitter.n_node_samples_disease
        cdef SIZE_t n_node_samples_control = splitter.n_node_samples_control
        
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double diff_impurity
        cdef double impurity_disease
        cdef double impurity_control
        cdef double objective
        cdef double original_impurity_disease
        cdef double original_impurity_control
        cdef double improvement_disease
        cdef double improvement_control
        cdef double differential_improvement
        
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record
        
        with nogil:

            rc = stack.push(0, n_node_samples_disease, 0, n_node_samples_control, -1, _TREE_UNDEFINED, 0, -INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY)
        
            
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                
                stack.pop(&stack_record)

                # Get data for the current node
                start_d = stack_record.start_d
                end_d = stack_record.end_d
                start_c = stack_record.start_c
                end_c = stack_record.end_c
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
            
                # Stack should contain the impurity of all samples reaching current node
                diff_impurity = stack_record.diff_impurity
                impurity_disease = stack_record.impurity_disease
                impurity_control = stack_record.impurity_control
                objective = stack_record.objective
                original_impurity_disease = stack_record.original_impurity_disease
                original_impurity_control = stack_record.original_impurity_control
                improvement_disease = stack_record.improvement_disease
                improvement_control = stack_record.improvement_control
                differential_improvement = stack_record.differential_improvement
                
                # NODE_RESET
                # Reset splitter on node samples[start:end]
                b = splitter.node_reset(start_d, end_d, start_c, end_c)
                
                n_node_samples_disease = end_d - start_d
                n_node_samples_control = end_c - start_c


				
                # Terminal node
                # Before splitting current node                
                is_leaf = (depth >= self.max_depth or
                           splitter.n_node_samples_disease < self.min_samples_split or
                           splitter.n_node_samples_disease  < 2 * self.min_samples_leaf or 
                           splitter.n_node_samples_control  < 2 * self.min_samples_leaf or
                           splitter.n_node_samples_control < self.min_samples_split)
						   
                if first:
                    original_impurity_disease = splitter.node_impurity_disease()
                    original_impurity_control = splitter.node_impurity_control()
                    
                    first = 0
                    
                    
                if not is_leaf:
                    
                    # Added for splitter_limit
                    splitter.node_split(start_d, end_d, start_c, end_c, original_impurity_disease, original_impurity_control, &split)
                    # if n_stage <= 5:
                        
                    #     splitter.node_split(start_d, end_d, start_c, end_c, original_impurity_disease, original_impurity_control, &split, features_bowl)
                        
                    # else:
                        
                    #     splitter.node_split(start_d, end_d, start_c, end_c, original_impurity_disease, original_impurity_control, &split, features_bowl)
                    
                if split.feature == -2:
                    
                    is_leaf = 1
                    
                node_id = self.add_node(parent, is_left, is_leaf, split.feature, depth,
                                split.threshold_d, split.threshold_c,
                                split.objective,  split.diff_impurity,
                                original_impurity_disease, original_impurity_control,
                                split.improvement_disease, split.improvement_control,
                                split.differential_improvement, split.differential_improvement,
                                split.impurity_disease, split.impurity_disease_left, split.impurity_disease_right,
                                split.impurity_control, split.impurity_control_left, split.impurity_control_right,
                                n_node_samples_disease, n_node_samples_control)
    
    
                splitter.node_value(self.value + node_id * self.value_stride)
                
                if not is_leaf:
                
                    # Push the right child on stack
                    rc = stack.push(split.pos_d, splitter.end_d, split.pos_c, splitter.end_c,
                                    depth + 1, node_id, 0, split.diff_impurity, 
                                    split.impurity_disease, split.impurity_control, 
                                    split.impurity_disease_right, split.impurity_control_right, 
                                    split.improvement_disease, split.improvement_control, 
                                    split.differential_improvement,
                                    split.objective)                    
                    
                    if rc == -1:
                        break
                            

                    # Push left child on stack
                    rc = stack.push(splitter.start_d, split.pos_d, splitter.start_c, split.pos_c, 
                                    depth + 1, node_id, 1, split.diff_impurity, 
                                    split.impurity_disease, split.impurity_control, 
                                    split.impurity_disease_left, split.impurity_control_left, 
                                    split.improvement_disease, split.improvement_control, 
                                    split.differential_improvement,
                                    split.objective)
                    
                    if rc == -1:
                        break
                        
                    
                if depth > max_depth_seen:
                    max_depth_seen = depth
                    
            if rc >= 0:
                rc = self._resize_c(self.node_count)

            if rc >= 0:
                self.max_depth = max_depth_seen
                
        if rc == -1:
            raise MemoryError()
                    
                
    cdef np.ndarray _get_value_ndarray(self):
        
        """Wraps value as a 3-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr
    
    
    cdef inline np.ndarray _apply_dense(self, object X):
        
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                              % type(X))
            
        # Extract input
        cdef const double[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # # Initialize auxiliary data-structure
        cdef Node node
        cdef SIZE_t i = 0
        cdef SIZE_t i_node = 0

        with nogil:
            
            for i in range(n_samples):

                i_node = 0
                    
                node = self.nodes[i_node]
                
                while node.feature != -2:
                    
                    node = self.nodes[i_node]
                    
                    if node.feature == -2:
                        break

                    if X_ndarray[i, node.feature] <= node.threshold_d:
                    
                        i_node = node.left_child
                        left = self.nodes[i_node]

                        
                        if left.feature == -2:
                            break

                    else:
                        i_node = node.right_child
                        right = self.nodes[i_node]

                        if right.feature == -2:
                            break

                out_ptr[i] = <SIZE_t>(i_node)  # node offset
                
        return out
    
    
    cpdef np.ndarray predict(self, object X):
        
        """Predict target for X."""
        
        out = self._get_value_ndarray().take(self._apply_dense(X), axis=0,
                                              mode='clip')

        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
    

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
            double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()
                
    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        
        """Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity), 0,
                    (capacity - self.capacity) *  sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0