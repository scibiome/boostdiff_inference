# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


from libc.math cimport fabs
import numpy as np
# import math
cimport numpy as np
cimport cython
import pandas as pd

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.math cimport floor, ceil, round

import random

from boostdiff.differential_trees.utils cimport log
from boostdiff.differential_trees.utils cimport rand_int
from boostdiff.differential_trees.utils cimport rand_uniform
from boostdiff.differential_trees.utils cimport RAND_R_MAX
from boostdiff.differential_trees.utils cimport safe_realloc


cdef double INFINITY = np.inf

from cpython cimport array
import array

import random
from random import randrange

cdef extern from "<math.h>":
    DTYPE_t sqrt(double m)
    

cdef inline void _init_split(SplitRecord* self, SIZE_t start_d, SIZE_t pos_d, SIZE_t end_d,
                             SIZE_t start_c, SIZE_t pos_c, SIZE_t end_c) nogil:

    self.feature = <SIZE_t>-2
    self.pos_d = start_d
    self.start_d = start_d
    self.end_d = end_d
    self.start_c = start_c
    self.pos_c = start_c
    self.end_c = end_c
    self.threshold_d = 0.
    self.threshold_c = 0.
    self.diff_impurity = -INFINITY
    self.impurity_disease = INFINITY
    self.impurity_control = INFINITY
    self.impurity_disease_left = INFINITY
    self.impurity_disease_right = INFINITY
    self.impurity_control_left = INFINITY
    self.impurity_control_right = INFINITY
    self.objective = -INFINITY
    self.original_impurity_disease = INFINITY
    self.original_impurity_control = INFINITY
    self.improvement_disease = -INFINITY
    self.improvement_control = -INFINITY
    self.differential_improvement = -INFINITY
    
    
cdef class Splitter:
        
    """
    Initialize the splitter.
    Take in the input data X, the target Y, and optional sample weights.
    """    
    
    def __cinit__(self, object random_state, SIZE_t max_features):

        
        self.samples_disease = NULL
        self.samples_control = NULL
        
        self.random_state = random_state
        
        self.features = NULL
        self.feature_values_disease = NULL
        self.feature_values_control = NULL
        
        self.n_features = 0
        
        self.start_d = 0
        self.end_d = 0
        self.pos_d = 0
        
        self.start_c = 0
        self.end_c = 0
        self.pos_c = 0    

        self.sum_total_disease = 0.0
        self.sum_left_disease = 0.0
        self.sum_right_disease = 0.0
        self.sq_sum_total_disease = 0.0
        self.sq_sum_left_disease = 0.0
        self.sq_sum_right_disease = 0.0
        
        self.sum_total_control = 0.0
        self.sum_left_control = 0.0
        self.sum_right_control = 0.0
        self.sq_sum_total_control = 0.0
        self.sq_sum_left_control = 0.0
        self.sq_sum_right_control = 0.0
        
        self.n_left_d = 0
        self.n_right_d = 0
        self.n_left_c = 0
        self.n_right_c = 0
        
        self.n_node_samples_disease = 0
        self.n_node_samples_control = 0
        
        self.max_features = max_features
        

    def __dealloc__(self):
        
        """Destructor."""

        free(self.samples_disease)
        free(self.samples_control)
        free(self.features)
        free(self.feature_values_disease)
        free(self.feature_values_control)

        
    def __getstate__(self):
        return {}


    def __setstate__(self, d):
        pass


    cdef int init(self, object X_disease, object X_control, double[:] y_disease, double[:] y_control) except -1:
        
        """Initialize the splitter.
        
        Take in the input data X_disease, X_control, the targets y_disease, y_control
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        
        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : ndarray, dtype=double
            This is the vector of targets, or true labels, for the samples
        """
        
        # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        # self.rand_r_state = np.random.RandomState(seed)
        self.rand_r_state = random.randint(0, RAND_R_MAX)
        
        cdef SIZE_t n_samples_disease = X_disease.shape[0]
        cdef SIZE_t n_samples_control = X_control.shape[0]
        
        cdef SIZE_t* samples_disease = safe_realloc(&self.samples_disease, n_samples_disease)
        cdef SIZE_t* samples_control = safe_realloc(&self.samples_control, n_samples_control)
        
        cdef SIZE_t n_features = X_disease.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)
        
        # Added for splitter_limit
        # cdef SIZE_t* features_sofar = safe_realloc(&self.features, n_features)
        
        self.n_samples_disease = n_samples_disease
        self.n_samples_control = n_samples_control
        
        for i in range(n_features):
            features[i] = i
            
        self.n_features = n_features
        
        # DISEASE samples
        for i in range(self.n_samples_disease):
            samples_disease[i] = i
        
        # CONTROL samples
        for i in range(self.n_samples_control):
            samples_control[i] = i
            
        self.X_disease = X_disease
        self.X_control = X_control
        
        self.y_disease = y_disease
        self.y_control = y_control
        
        self.n_node_samples_disease = n_samples_disease
        self.n_node_samples_control = n_samples_control
        
        safe_realloc(&self.feature_values_disease, n_samples_disease)
        safe_realloc(&self.feature_values_control, n_samples_control)
        
        
    cdef int node_reset(self, SIZE_t start_d, SIZE_t end_d, 
                        SIZE_t start_c, SIZE_t end_c) nogil except -1:
        
        """Reset splitter on node samples[start:end].
        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        
        Initialize the criterion at node samples[start:end] and
            children samples[start:start] and samples[start:end]."""
                   
        self.start_d = start_d
        self.end_d = end_d
        self.start_c = start_c
        self.end_c = end_c
        
        # Reset the sum totals and squared sum totals
        self.sum_total_disease = 0.0
        self.sum_total_control = 0.0
        
        self.sq_sum_total_disease = 0.0
        self.sq_sum_total_control = 0.0
        
        cdef SIZE_t p_d, p_c, i
        cdef double y_i_disease, y_i_control
        
        # DISEASE dataset
        for p_d in range(self.start_d, self.end_d):
            
            i = self.samples_disease[p_d]
            y_i_disease = self.y_disease[i]
            self.sum_total_disease += y_i_disease
            self.sq_sum_total_disease += y_i_disease * y_i_disease
        
        self.n_node_samples_disease = end_d - start_d

        # Reset to pos=start
        self.reset_disease()

        # CONTROL dataset
        for p_c in range(self.start_c, self.end_c):
            
            i = self.samples_control[p_c]
            y_i_control = self.y_control[i]
            self.sum_total_control += y_i_control
            self.sq_sum_total_control += y_i_control * y_i_control

        self.n_node_samples_control = end_c - start_c
        
        # Reset to pos=start
        self.reset_control()
        
        
        return 0
        
        
    cdef int reset_disease(self) nogil except -1:

        """Reset the criterion at pos=start.
        """

        self.sum_left_disease = 0.0
        self.sum_right_disease = self.sum_total_disease
        self.sq_sum_left_disease = 0.0
        self.sq_sum_right_disease = self.sq_sum_total_disease
        
        self.n_left_d = 0
        self.n_right_d = self.n_node_samples_disease
        self.pos_d = self.start_d
        
        return 0
    
    
    cdef int reset_control(self) nogil except -1:

        """Reset the criterion at pos=start.
        """

        self.sum_left_control = 0.0
        self.sum_right_control = self.sum_total_control
        self.sq_sum_left_control = 0.0
        self.sq_sum_right_control = self.sq_sum_total_control
        
        self.n_left_c = 0
        self.n_right_c = self.n_node_samples_control
        self.pos_c = self.start_c
        
        return 0
    
    
    cdef int reverse_reset_disease(self) nogil except -1:

        """Reset the criterion at pos=end.
        """
        
        self.sum_right_disease = 0.0
        self.sum_left_disease = self.sum_total_disease
        self.sq_sum_right_disease = 0.0
        self.sq_sum_left_disease = self.sq_sum_total_disease
        
        self.n_right_d = 0
        self.n_left_d = self.n_node_samples_disease
        self.pos_d = self.end_d
        
        return 0
  
      
    cdef int reverse_reset_control(self) nogil except -1:

        """Reset the criterion at pos=end
        """
        
        self.sum_right_control = 0.0
        self.sum_left_control = self.sum_total_control
        self.sq_sum_right_control = 0.0
        self.sq_sum_left_control = self.sq_sum_total_control

        self.n_right_c = 0
        self.n_left_c = self.n_node_samples_control
        self.pos_c = self.end_c
        
        return 0
    
    
    cdef int update_disease(self, SIZE_t new_pos_d) nogil except -1:
        
        """
        Updated statistics by moving samples[pos:new_pos] to the left.
        Updates self.sq_sum_left and self.sq_sum_right
        """
        
        cdef SIZE_t p, i

        cdef SIZE_t pos_d = self.pos_d
        cdef SIZE_t end_d = self.end_d
        
        cdef double y_i
        
        cdef SIZE_t* samples_disease = self.samples_disease

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos_d - pos_d) <= (end_d - new_pos_d):
            
            for p in range(pos_d, new_pos_d):
                i = samples_disease[p]
                y_i = self.y_disease[i]
                self.sum_left_disease += y_i
                self.sq_sum_left_disease += y_i * y_i
                self.n_left_d += 1
        else:
            
            self.reverse_reset_disease()

            for p in range(end_d - 1, new_pos_d - 1, -1):
                i = samples_disease[p]
                y_i = self.y_disease[i]
                self.sum_left_disease -= y_i
                self.sq_sum_left_disease -= y_i * y_i
                self.n_left_d -= 1
      
        self.n_right_d = (self.n_node_samples_disease - self.n_left_d)
        self.sum_right_disease = self.sum_total_disease - self.sum_left_disease
        self.sq_sum_right_disease = self.sq_sum_total_disease - self.sq_sum_left_disease
        self.pos_d = new_pos_d
                
        return 0
    
        
    cdef int update_control(self, SIZE_t new_pos_c) nogil except -1:
        
        """
        Updated statistics by moving samples[pos:new_pos] to the left.
        """

        cdef SIZE_t p, i
        
        cdef SIZE_t pos_c = self.pos_c
        cdef SIZE_t end_c = self.end_c
        
        cdef double sum_left_control = self.sum_left_control
        cdef double sum_right_control = self.sum_right_control
        cdef double sum_total_control = self.sum_total_control  
        
        cdef double y_i
        
        cdef SIZE_t* samples_control = self.samples_control

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos_c - pos_c) <= (end_c - new_pos_c):
            
            for p in range(pos_c, new_pos_c):
                i = samples_control[p]
                y_i = self.y_control[i]
                self.sum_left_control += y_i
                self.sq_sum_left_control += y_i * y_i
                self.n_left_c += 1
        else:
            
            self.reverse_reset_control()

            for p in range(end_c - 1, new_pos_c - 1, -1):
                i = self.samples_control[p]
                y_i = self.y_control[i]
                self.sum_right_control -= y_i
                self.sq_sum_left_control -= y_i * y_i
                self.n_left_c -= 1

        self.n_right_c = (self.n_node_samples_control - self.n_left_c)
        self.sum_right_control = self.sum_total_control - self.sum_left_control
        self.sq_sum_right_control = self.sq_sum_total_control - self.sq_sum_left_control
        self.pos_c = new_pos_c       
        
        return 0
   
    
    cdef double node_impurity_disease(self) nogil:
        
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        
        cdef SIZE_t i
        cdef double impurity_disease
        
        impurity_disease = self.sq_sum_total_disease / self.n_samples_disease
        impurity_disease -= (self.sum_total_disease / self.n_samples_disease) ** 2.0
        
        return impurity_disease
    
    
    cdef double node_impurity_control(self) nogil:
        
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        
        cdef SIZE_t i
        cdef double impurity_control

        impurity_control = self.sq_sum_total_control / self.n_samples_control
        impurity_control -= (self.sum_total_control / self.n_samples_control) ** 2.0
        
        return impurity_control
    
    
    cdef int children_impurity_improvement_disease_mse(self, SIZE_t start_d, SIZE_t pos_d, SIZE_t end_d, double impurity_parent_disease, double *improvement_disease, double *mse_disease, double *mse_disease_left, double *mse_disease_right) nogil except -1:
        
        """
        
        Evaluate the MSE impurity in children nodes (sum of the left and right impurities),
        i.e. the impurity of the left child (samples[start:pos]) 
        and the impurity the right child (samples[pos:end]).
        """
        
        # Calculations for the DISEASE dataset
        cdef SIZE_t* samples = self.samples_disease
        cdef SIZE_t n_left_d = pos_d - start_d
        cdef SIZE_t n_right_d = end_d - pos_d        
        cdef SIZE_t p, i
        cdef double y_i
        cdef double fraction_disease
        cdef double fraction_disease_left
        cdef double fraction_disease_right
        
        mse_disease_left[0] = self.sq_sum_left_disease / self.n_left_d
        mse_disease_right[0] = self.sq_sum_right_disease / self.n_right_d

        mse_disease_left[0] -= (self.sum_left_disease / self.n_left_d) ** 2.0
        mse_disease_right[0] -= (self.sum_right_disease / self.n_right_d) ** 2.0

        mse_disease[0] = mse_disease_left[0] + mse_disease_right[0]
        
        
        fraction_disease = <double>self.n_node_samples_disease / <double>self.n_samples_disease
        fraction_disease_left = <double>self.n_left_d / <double>self.n_node_samples_disease
        fraction_disease_right = <double>self.n_right_d / <double>self.n_node_samples_disease
        improvement_disease[0] =  (fraction_disease) * (impurity_parent_disease - (fraction_disease_left*mse_disease_left[0]) - (fraction_disease_right*mse_disease_right[0]))

        return 0
    
    
    cdef int children_impurity_improvement_control_mse(self, SIZE_t start_c, SIZE_t pos_c, SIZE_t end_c, double impurity_parent_control, double *improvement_control,  double *mse_control, double *mse_control_left, double *mse_control_right) nogil except -1:
        
        """
        Evaluate the MSE impurity in children nodes (sum of the left and right impurities),
        i.e. the impurity of the left child (samples[start:pos])
        and the impurity the right child (samples[pos:end]).
        """

        # Calculations for the CONTROL dataset
        cdef SIZE_t* samples_control = self.samples_control
        cdef SIZE_t n_left_c = pos_c - start_c
        cdef SIZE_t n_right_c = end_c - pos_c

        cdef SIZE_t p, i
        cdef double y_i_control
        cdef double fraction_control
        cdef double fraction_control_left
        cdef double fraction_control_right
        
        mse_control_left[0] = self.sq_sum_left_control / self.n_left_c
        mse_control_right[0] = self.sq_sum_right_control / self.n_right_c

        mse_control_left[0] -= (self.sum_left_control / self.n_left_c) ** 2.0
        mse_control_right[0] -= (self.sum_right_control / self.n_right_c) ** 2.0
        
        mse_control[0] = mse_control_left[0] + mse_control_right[0]
        
        fraction_control = <double>self.n_node_samples_control / <double>self.n_samples_control
        fraction_control_left = <double>self.n_left_c / <double>self.n_node_samples_control
        fraction_control_right = <double>self.n_right_c / <double>self.n_node_samples_control
        improvement_control[0] = (fraction_control) * (impurity_parent_control - (fraction_control_left*mse_control_left[0]) - (fraction_control_right*mse_control_right[0]))
        
        return 0
    

    cdef int find_thresholds_disease(self, SIZE_t current_feature, double impurity_parent_disease, SplitRecord* current_best) nogil except -1:
        
        """
        Find the best disease threshold for current feature
        """

        # Create the SplitRecord to store if the stats are better
        cdef SplitRecord best, current
        
        # than the current_best_impurity
        cdef double best_improvement_disease = -INFINITY
        
        cdef SIZE_t i
                
        cdef SIZE_t start_d = self.start_d
        cdef SIZE_t end_d = self.end_d
        
        _init_split(&best, self.start_d, self.pos_d, self.end_d, self.start_c, self.pos_c, self.end_c)
        
        # Initializations
        cdef double* Xf_disease = self.feature_values_disease
        cdef SIZE_t* samples_disease = self.samples_disease
        
        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        
        # Disease
        for i in range(start_d, end_d):
            Xf_disease[i] = self.X_disease[samples_disease[i], current_feature]
        
        sort(Xf_disease + start_d, samples_disease + start_d, end_d - start_d)

        # For this new feature
        self.reset_disease()
        
        # Start value from the disease dataset
        cdef SIZE_t p_d = start_d
        
        # Added for RMSE
        cdef double mse_disease = INFINITY
        cdef double mse_disease_left = INFINITY
        cdef double mse_disease_right = INFINITY
        cdef double improvement_disease =  -INFINITY

        # While loop for the DISEASE dataset
        while p_d < end_d:
            
            p_d += 1
            
            # Reject if min_samples_leaf is not guaranteed
            if (((p_d - start_d) < self.min_samples_leaf) or
                ((end_d - p_d) < self.min_samples_leaf)):
                continue
            
            self.update_disease(p_d)
            
            # Added for RMSE
            current_impurity_disease_res = self.children_impurity_improvement_disease_mse(start_d, p_d, end_d, impurity_parent_disease, &improvement_disease, &mse_disease, &mse_disease_left, &mse_disease_right)

            if improvement_disease > best_improvement_disease:
            
                best_improvement_disease = improvement_disease

                # sum of halves is used to avoid infinite value
                if p_d != start_d:
                    current_best.threshold_d = (Xf_disease[p_d - 1] / 2.0) + (Xf_disease[p_d] / 2.0)
                    
                    if ((current_best.threshold_d == Xf_disease[p_d]) or
                        (current_best.threshold_d == INFINITY) or
                        (current_best.threshold_d == -INFINITY)):
                        current_best.threshold_d = Xf_disease[p_d - 1]
                        
                    current_best.pos_d = p_d
                    current_best.start_d = start_d
                    current_best.end_d = end_d
                    current_best.impurity_disease = mse_disease
                    current_best.impurity_disease_left = mse_disease_left
                    current_best.impurity_disease_right = mse_disease_right
                    current_best.feature = current_feature
                    current_best.improvement_disease = best_improvement_disease
                
        return 0


    cdef int find_thresholds_control(self, SIZE_t current_feature, double impurity_parent_control, SplitRecord* current_best_control) nogil except -1:
        
        """
        Find the best disease threshold for current feature
        """
  
        # Create the SplitRecord to store if the stats are better
        cdef SplitRecord best, current
        
        cdef SIZE_t i     
               
        # For tracking the best impurity for current feature
        # Used only within the function
        # cdef DTYPE_t best_impurity = current_best.impurity_control
        cdef DTYPE_t best_improvement_control  = -INFINITY
        
        cdef SIZE_t start_c = current_best_control.start_c
        cdef SIZE_t end_c = current_best_control.end_c
        
        _init_split(&best, self.start_d, self.pos_d, self.end_d, self.start_c, self.pos_c, self.end_c)
        
        # Initializations
        cdef double* Xf_control = self.feature_values_control
        cdef SIZE_t* samples_control = self.samples_control
        
        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        # Disease
        for i in range(start_c, end_c):
            Xf_control[i] = self.X_control[samples_control[i], current_feature]
           

        sort(Xf_control + start_c, samples_control + start_c, end_c - start_c)
        
        
        # Evaluate all splits
        self.reset_control()  
        
        # Start value from the control dataset
        cdef SIZE_t p_c = start_c
        
        # Added for RMSE
        cdef double rmse_control = INFINITY
        cdef double rmse_control_left = INFINITY
        cdef double rmse_control_right = INFINITY
        cdef double improvement_control = INFINITY

        # While loop for the CONTROL dataset
        # Go through all values for the current feature
        while p_c < end_c:
            
            p_c += 1
            # Reject if min_samples_leaf is not guaranteed
            if (((p_c - start_c) < self.min_samples_leaf) or
                ((end_c - p_c) < self.min_samples_leaf)):
                continue
                
            self.update_control(p_c)
            
            current_impurity_control_res = self.children_impurity_improvement_control_mse(start_c, p_c, end_c, impurity_parent_control, &improvement_control, &rmse_control, &rmse_control_left, &rmse_control_right)


            # if current_impurity_control < best_impurity:
            if improvement_control > best_improvement_control:
                
                best_improvement_control = improvement_control
                
                # sum of halves is used to avoid infinite value
                current_best_control.threshold_c = Xf_control[p_c - 1] / 2.0 + Xf_control[p_c] / 2.0

                if ((current_best_control.threshold_c == Xf_control[p_c]) or
                    (current_best_control.threshold_c == INFINITY) or
                    (current_best_control.threshold_c == -INFINITY)):
                    
                    current_best_control.threshold_c = Xf_control[p_c - 1]
                    
     
                current_best_control.pos_c = p_c
                current_best_control.start_c = start_c
                current_best_control.end_c = end_c
                current_best_control.impurity_control = rmse_control
                current_best_control.impurity_control_left = rmse_control_left
                current_best_control.impurity_control_right = rmse_control_right
                current_best_control.feature = current_feature
                current_best_control.improvement_control = best_improvement_control
  
        return 0
    
    
    cdef void node_value(self, double* value) nogil:
        
        """Compute the node value of samples[start:end] into dest."""
        
        value[0] = self.sum_total_disease / self.n_node_samples_disease
        
        
    cdef int node_split(self, SIZE_t start_d, SIZE_t end_d, SIZE_t start_c, SIZE_t end_c, double impurity_parent_disease, double impurity_parent_control, 
                        SplitRecord* split
                        ) nogil except -1:
        
        """
        Step 1: For each feature, find the best threshold for disease and control datasets separately
        Step 2: For the two best thresholds, calculate the differential impurity
        Step 3: Iterate Steps 1 and 2 until you get the feature-threshold_d-threshold_d combination
                with the highest differential impurity
        """
            
        cdef SplitRecord current_disease, current_control, best
    
        self.start_d = start_d
        self.end_d = end_d
        self.start_c = start_c
        self.end_c = end_c
        
        cdef SIZE_t f_i = self.n_features
        
        # Initialize record values
        _init_split(&best, self.start_d, self.pos_d, self.end_d, self.start_c, self.pos_c, self.end_c)
        _init_split(&current_disease, self.start_d, self.pos_d, self.end_d, self.start_c, self.pos_c, self.end_c)
        _init_split(&current_control, self.start_d, self.pos_d, self.end_d, self.start_c, self.pos_c, self.end_c)
        # Features
        cdef SIZE_t* features = self.features
        cdef SIZE_t n_visited_features = 0
        
        cdef UINT32_t* random_state = &self.rand_r_state
        
        cdef double best_diff_impurity = best.diff_impurity
        cdef double current_differential_improvement
        cdef double current_objective
            
        cdef SIZE_t f_j
        cdef SIZE_t current_feature
        
        cdef double a
        
        cdef double temp_diffimp
    
        # Keep on finding a feature with good thresholds for disease and control
        # Until max_features has been reached
        while (n_visited_features < self.max_features):

            
            # # Draw a feature at random
            f_j = rand_int(0, self.n_features, random_state)
                
            # Previous code
            current_feature = features[f_j]
            
            # Update no. of visited features
            n_visited_features += 1
            
            if current_feature < self.n_features:
            
                # Find the best feature threshold from the disease dataset
                current_disease_res = self.find_thresholds_disease(current_feature, impurity_parent_disease, &current_disease)
                
                # Find the best feature threshold from the control dataset
                current_control_res = self.find_thresholds_control(current_feature, impurity_parent_control, &current_control)

                # IMPORTANT: For each feature should find a valid threshold for both the disease and control datasets                
                if current_disease.start_d != current_disease.pos_d and current_control.start_c != current_control.pos_c:

                    current_differential_improvement = current_disease.improvement_disease - current_control.improvement_control
                    
                    # It should satisfy the min samples leaf
                    if (current_control.pos_c - current_control.start_c) >= self.min_samples_leaf and \
                    (current_disease.pos_d - current_disease.start_d) >= self.min_samples_leaf: 
                            
                        temp_diffimp = <double>best.differential_improvement
                        
                        if current_differential_improvement > 0 and current_differential_improvement > temp_diffimp:

                            # Merge the best result from the current_disease and current_control
                            best.feature = current_feature
                            best.threshold_d = current_disease.threshold_d
                            best.start_d = current_disease.start_d
                            best.pos_d = current_disease.pos_d
                            best.end_d = current_disease.end_d
							
                            # Store the original impurity values
                            best.impurity_disease = current_disease.impurity_disease
                            best.impurity_control = current_control.impurity_control
                            best.impurity_disease_left = current_disease.impurity_disease_left
                            best.impurity_disease_right = current_disease.impurity_disease_right
                            best.impurity_control_left = current_control.impurity_control_left
                            best.impurity_control_right = current_control.impurity_control_right
                            best.original_impurity_disease = current_disease.original_impurity_disease
                            best.original_impurity_control = current_control.original_impurity_control
                            best.improvement_disease = current_disease.improvement_disease
                            best.improvement_control = current_control.improvement_control
                            best.differential_improvement = current_differential_improvement
                            best.threshold_c = current_control.threshold_c
                            best.start_c = current_control.start_c
                            best.end_c = current_control.end_c
                            best.pos_c = current_control.pos_c

        cdef SIZE_t partition_end_d, partition_end_c, p_d, p_c        
        cdef SIZE_t* samples_disease = self.samples_disease
        cdef SIZE_t* samples_control = self.samples_control
       
        # Reorganize disease dataset into samples[start:best.pos] + samples[best.pos:end]
        if best.pos_d < end_d:

            p_d, partition_end_d = start_d, end_d

            while p_d < partition_end_d:
                
                if self.X_disease[samples_disease[p_d], best.feature] <= best.threshold_d:
                    p_d += 1
                else:
                    # Keep on swapping
                    partition_end_d -= 1
                    samples_disease[p_d], samples_disease[partition_end_d] = samples_disease[partition_end_d], samples_disease[p_d]

        self.reset_disease() # Reset pos_d
        self.update_disease(best.pos_d) # Update statistics

        if best.pos_c < end_c:
            
            p_c, partition_end_c = start_c, end_c

            while p_c < partition_end_c:
                # samples[p] is the index of the data point
                if self.X_control[samples_control[p_c], best.feature] <= best.threshold_c:
                    p_c += 1
                else:
                    partition_end_c -= 1
                    samples_control[p_c], samples_control[partition_end_c] = samples_control[partition_end_c], samples_control[p_c]
                
        self.reset_control()
        self.update_control(best.pos_c)

        split[0] = best
        return 0    
    
# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(double* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(double* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline double median3(double* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef double a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(double* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(double* Xf, SIZE_t* samples,
                            SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(double* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1
