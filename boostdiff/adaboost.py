
from .differential_trees.diff_tree import DiffTree
from .differential_trees.splitter import Splitter
import numpy as np
import warnings
import numbers


def check_random_state(seed):
    
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
    
def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def r2_score(y_true, y_pred, *, sample_weight=None):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg)
        return float("nan")

    if sample_weight is not None:
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        weight * (y_true - np.average(y_true, axis=0)) ** 2
    ).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    
    if valid_score:
        output_score = 1 - (numerator / denominator)
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    # output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return output_score



class AdaBoostDiffRegressor():
    

    def __init__(self, base_estimator=DiffTree,
                 min_samples_leaf = 2, min_samples_split = 6,
                 max_depth = 2, max_features = 40, n_estimators=50,
                 learning_rate=1.0, 
                 loss="square", variable_importance = "disease_importance",
                 random_state=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
    
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        
        self.learning_rate = learning_rate
        self.loss = loss
        
        self.variable_importance = variable_importance
        self.random_state = random_state
        self.estimator_count = 0
        
        self.errors_disease = []
        self.errors_control = []      
    
    
    def _boost(self, iboost, X_disease, X_control, output_disease, output_control, sample_weight, n_subsample, random_state):
                
        n_samples_disease = X_disease.shape[0]
        n_samples_control = X_control.shape[0]

        # random_state = np.random.RandomState(iboost)
        # random_state = check_random_state(self.random_state)
        # print("random_state", random_state)        
        
        estimator =  DiffTree(self.min_samples_leaf, self.min_samples_split, 
                          n_samples_disease, n_samples_control,
                          self.max_depth, self.max_features)

        bootstrap_idx_dis = np.random.choice(n_samples_disease, n_subsample, replace=True, p=sample_weight)
        bootstrap_idx_con = np.random.choice(n_samples_control, n_subsample, replace=True)

        X_dis = X_disease.iloc[bootstrap_idx_dis,:].values
        X_con = X_control.iloc[bootstrap_idx_con,:].values
        
        # Expression of target gene as predicted values
        y_disease = output_disease[bootstrap_idx_dis]
        y_control = output_control[bootstrap_idx_con]
        
        
        estimator.build(X_dis, X_con, y_disease, y_control, iboost)
        
        if self.variable_importance == "disease_importance":
            self.feature_importances[iboost, :] = estimator.get_variable_importance_disease_gain()

                
        elif self.variable_importance == "differential_improvement":
            self.feature_importances[iboost, :] = estimator.get_variable_importance_differential_improvement()

        y_predict = estimator.predict(X_disease.values).flatten()

        error_vect = np.absolute(y_predict - output_disease)

        sample_mask = sample_weight > 0

        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
                   
        error_max = masked_error_vector.max()

        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == "square":
            masked_error_vector **= 2
        elif self.loss == "exponential":
            masked_error_vector = 1.0 - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        
        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
                
            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)
        
        if not iboost == self.n_estimators - 1:
            
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate)

        self.estimators_.append(estimator)
        self.estimator_count +=1        
        
        y_predict_append = self.predict(X_disease.values).flatten()
        y_predict_append_con = self.predict(X_control.values).flatten()
            
        error_mean_append = np.absolute(y_predict_append - output_disease)

        error_mean_append_con = np.absolute(y_predict_append_con - output_control)
        

        self.errors_disease.append(np.mean(error_mean_append))
        self.errors_control.append(np.mean(error_mean_append_con))
                
        return sample_weight, estimator_weight, estimator_error


    def _get_median_predict(self, X, limit):
        
        n_samples = X.shape[0]
        
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T[0]
  
        # Sorts the samples for each estimator
        sorted_idx = np.argsort(predictions, axis=1)
        
        # SHould be along the rows
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        

        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        
        median_estimators = sorted_idx[np.arange(n_samples), median_idx]

        return predictions[np.arange(n_samples), median_estimators]


    def predict(self, X):

        
        return self._get_median_predict(X, self.n_estimators)


    def staged_predict(self, X):
        
        """Return staged predictions for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.
        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        Yields
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        """
        
        for i, _ in enumerate(self.estimators_, 1):
            
            yield self._get_median_predict(X, limit=i)
            
            
    def fit(self, X_disease, X_control, output_disease, output_control, n_subsample, sample_weight=None):
       
        """Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
        """
        
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # Put genes in the columns
        X_disease = X_disease.T
        X_control = X_control.T
        
        # Preprocessing of datasets
        n_disease = X_disease.shape[0]
        n_control = X_control.shape[0]
        
        n_input_genes = X_disease.shape[1]
        
        self.feature_importances = np.zeros([self.n_estimators, n_input_genes], dtype=np.float64)

        
        if sample_weight is None:
            sample_weight = np.ones(n_disease)
            
        sample_weight /= sample_weight.sum()
        
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)


        # Initializion of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)
        

        for iboost in range(self.n_estimators):
            
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X_disease, X_control, output_disease, output_control, sample_weight, n_subsample, random_state)

            # Early termination
            if sample_weight is None:
                break
        
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                
                print("estimator error is zero")
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            # Normalization of sample weights
            # After scaling disease samples only
            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        
        return self
            
            
    def staged_score(self, X, y, sample_weight=None):
        
        """Return staged scores for X, y.
        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            Labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Yields
        ------
        z : float
        """

        for y_pred in self.staged_predict(X):
            
            yield r2_score(y, y_pred, sample_weight=sample_weight)
            
         
    def calculate_adaboost_importance(self):
        
        """
        """
                
        norm = self.estimator_weights_.sum()
        
        final_importances = np.zeros((self.estimator_count, len(self.feature_importances[0])))
        
        for i in range(self.estimator_count):
            
            final_importances[i,:] = self.estimator_weights_[i] * self.feature_importances[i,:]

        
        final_importances = np.sum(final_importances, axis=0)
        final_importances = final_importances / norm

        return final_importances