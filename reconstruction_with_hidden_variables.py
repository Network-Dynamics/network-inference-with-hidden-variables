import numpy as np
from scipy.integrate import cumtrapz
from numpy.linalg import pinv, lstsq
from numba import njit
from sklearn import metrics as skl_metrics
from copy import deepcopy


def rescale(val, interval=(0, 2*np.pi)):
    """
    rescale val to interval, e.g. (-pi, pi) or (0, 2pi)
    """
    return (val - interval[0]) % (interval[1] - interval[0]) + interval[0]


@njit
def _calculate_otsu_threshold_index(p, nbins):
    """
    numba optimized calculation of threshold index,
    see otsu_threshold(...) for details
    """
    w0 = np.cumsum(p)
    w1 = 1 - w0
    muT = np.sum(np.arange(nbins) * p)
    muk = np.cumsum(np.arange(nbins) * p)

    # ignore first and last values to avoid zero division error
    mu0 = muk[1:-1] / w0[1:-1]
    mu1 = (muT - muk[1:-1]) / (1 - w0[1:-1])

    sigmab_square = w0[1:-1]*w1[1:-1]*(mu0 - mu1)**2
    # index of threshold, correction due to cutoff
    k1 = np.argmax(sigmab_square) + 1
    k2 = len(sigmab_square) - np.argmax(sigmab_square[::-1])
    k = (k1+k2)//2  # use bin which is centered on plateau
    # k = np.argmax(sigmab_square) + 1 # just use first bin
    return k


def otsu_threshold(vals, bins=128):
    """
    a simple algorithm to separate a bimodal distribution of values
    returns the optimal threshold separating the data

    Parameters:
    -----------
    vals: np.array
        values to calculate threshold of. values should be approximately
        bimodally distributed
    bins: int
        number of bins used for grouping of vals

    Returns:
    --------
    threshold: float
        optimal threshold maximizing the inter-class variance
    """
    # normalize values:
    n = len(vals)  # number of values
    vals = np.sort(vals)
    # create histogram of values
    hist, bin_edges = np.histogram(vals, bins)
    p = hist / n
    # calculate threshold index of corresponding bin
    k = _calculate_otsu_threshold_index(p, bins)
    # middle of optimal bin as threshold
    return (bin_edges[k] + bin_edges[k + 1]) / 2


def mean_abs_err_angles(phi1, phi2):
    """
    calculate mean absolute error of two angles
    """
    d = np.array(np.abs(rescale(phi1, interval=(0, 2*np.pi)) - rescale(phi2, interval=(0, 2*np.pi))))
    idx = d>np.pi
    d[idx] = 2*np.pi - d[idx]
    return np.mean(d)


@njit
def mae_threshold_njit(xreal, xpred, n):
    """ calculate mean absolute error """
    if xreal.shape != xpred.shape:
        raise ValueError("both inputs have to have the same shape")
    return np.sum(np.absolute(xreal - xpred)) / n


def mean_abs_err_thresholded_data(xreal, xpred):
    """
    calculate mean absolute error, but with previously thresholded data,
    so not all entries contribute to the error, but only the values larger than 0 in xreal
    """
    n = len(xreal[xreal != 0])
    return mae_threshold_njit(xreal, xpred, n)


class Reconstructor():
    """
    reconstruction using the direct method with incomplete information

    Parameters:
    -----------
    y: np.ndarray, shape: (N, M)
        The measured time series of y.
        Here, N is the dimension of the underlying network, and M is the number of data points.
    t: np.ndarray, shape: (M,)
        time instants of the measured data
    f, g: callable with f=f(t, y, **fkwargs), g=g(t, y, **gkwargs)
        the known dynamical system
        has to return np.array shape (N,) or (N, len(t))

        dot x_i(t) = f_i(t, y, **fkwargs)
        dydt_fac_i * dot y_i(t) = g_i(t, y, **gkwargs) + sum_j sin(x_j - x_i)
    fkwargs, gkwargs: dict
        keyword arguments of f and g, respectively
    dydt_fac, number or np.array with shape (N,)
        if dy/dt is multiplied by some factor, this factor has to be specified 
    threshold_func: str or callable, optional
        specifies thresholding function,
        currently accepts callable and 'otsu'.
        If callable, it has to accept parameters  x, **kwargs, with x being the values to calculate the
        threshold value of.
    directed: bool, optional
        whether the underlying network is directed.
        True, if dydt_fac is an array, since the reconstructed matrix then is not symmetric
    dxparams: dict, optional
        TODO: finish doc
    """
    def __init__(self, y, t, f, g, fkwargs, gkwargs, dydt_fac=1, threshold_func="otsu", threshold_kwargs={'bins':128}, directed=True, dxparams=None):
        self.y = y
        self.t = t
        self.f_eval = f(self.t, self.y, **fkwargs)
        self.g_eval = g(self.t, self.y, **gkwargs)
        self.N, self.M = self.y.shape
        if isinstance(dydt_fac, np.ndarray) and dydt_fac.shape==(self.N,):
            self.dydt_fac = dydt_fac[:, np.newaxis]
            self.directed = True
        else:
            self.dydt_fac = dydt_fac
            self.directed = directed
        self.deltax = self.calculate_deltax()

        self.set_threshold_function(threshold_func, threshold_kwargs)

        if dxparams is None:
            self.dxparams = {
                'method': "exp",  # which method to use for reconstruction of delta x
                'scale': True,  # if delta x should be rescaled to the interval
                'interval': (-np.pi, np.pi)
            }
        else:
            self.dxparams = dxparams

    def set_threshold_function(self, threshold, threshold_kwargs):
        if callable(threshold):
            self.threshold_func = lambda x: threshold(x, **threshold_kwargs)
        elif threshold == "otsu":
            self.threshold_func = lambda x: otsu_threshold(x, **threshold_kwargs)
        else:
            raise ValueError(f"invalid argument: {threshold}")

    def calculate_deltax(self):
        """ shape: (N, N, M)
        deltax[i, j]: TS of x_j - x_i
        """
        N, M = self.f_eval.shape
        return cumtrapz(np.array(
            [self.f_eval[j] - self.f_eval[i] for i in range(N) for j in range(N)]).reshape(N, N, M),
            self.t, initial=0, axis=2)

    def calculate_L(self, M=None):
        """ shape: (N, M) """
        if not M:
            M = self.M
        dot_y = np.gradient(self.y[:, :M], self.t[:M], axis=1)
        self.L = dot_y - self.g_eval[:, :M]

    def calculate_R(self):
        """ shape: (N, 2N, M) """
        self.R = np.zeros((self.N, 2*self.N, self.M))
        self.R[:, :self.N, :] = np.sin(self.deltax)
        self.R[:, self.N:, :] = np.cos(self.deltax)

    def calculate_Copt(self, M=None):
        """ shape: (N, 2N) """
        if not hasattr(self, 'L') or \
            (M is not None and self.L.shape[1] != M) or \
                (self.L.shape[1] != self.M):
            self.calculate_L(M)

        if not hasattr(self, 'R'):
            self.calculate_R()

        if M and M < self.M:  # use only M data points
            R = self.R[:, :, :M]
        else:  # use all data points
            R = self.R

        if self.method == 'lstsq':
            self.C_opt = np.array([lstsq(R[i].T, self.L[i].T, rcond=None)[0].T for i in range(self.N)])
        elif self.method == 'pinv':
            pinvR = pinv(R)
            self.C_opt = np.array([np.dot(self.L[i], pinvR[i]) for i in range(self.N)])
        else:
            raise ValueError(f"Invalid optimization method: {self.method}")

    def reconstruct_delta_x0(self, i, j):
        """
        calculate initial phase differences
        the reconstructed phase differences lie within the interval [-pi, pi)

        Parameters:
        -----------
        i, j: int
            select edge j->i

        Returns:
        --------
        float
            initial phase difference at K_ij, i.e. edge j->i, which gives x_ji_0

        Raises:
        -------
        ValueError: if 'method' is not in ['exp', 'arctan2']
        """
        if self.directed:
            if self.dxparams['method'] == "arctan2":
                delta_x0_ji = np.arctan2(self.C_opt[i, j+self.N], self.C_opt[i, j])
            elif self.dxparams['method'] == "exp":
                delta_x0_ji = (-1j*np.log((self.C_opt[i, j] + 1j*self.C_opt[i, j+self.N])/self.K_rec[i, j])).real
                delta_x0_ji = rescale(delta_x0_ji, (-np.pi, np.pi))
            else:
                raise ValueError(f"Invalid argument passed to method", self.dxparams['method'])
        else:
            if self.dxparams['method'] == "arctan2":
                delta_x0_ji = np.arctan2(
                    self.C_opt[i, j+self.N] - self.C_opt[j, i+self.N], self.C_opt[i, j] + self.C_opt[j, i])
            elif self.dxparams['method'] == "exp":
                delta_x0_ji = (-1j*np.log(
                    ((self.C_opt[i, j] + self.C_opt[j, i])/2 + 1j * (self.C_opt[i, j+self.N] - self.C_opt[j, i+self.N])/2) /
                        (self.K_rec[i, j] + self.K_rec[j, i])/2)).real
                delta_x0_ji = rescale(delta_x0_ji, (-np.pi, np.pi))
            else:
                raise ValueError(f"Invalid argument passed to method", self.dxparams['method'])
        return delta_x0_ji

    def reconstruct_delta_x(self, int_delta_f):
        """
        returns a dictionary d containing the reconstructed intital phase difference.
        For each node i all incident edge weights are tested if their value is larger than threshold
        If true, the phase difference is calculated.

        TODO: check doc

        Parameters:
        -----------
        int_delta_f: np.ndarray, shape: (N, N)
            matrix of each integral over all differences of f(y_i, t) and f(y_j, t) from t=0 to t->infty.

        Returns:
        --------
        list of tuples ((j, i), delta_x)
            delta x is the difference of x in the fixed point at edge j->i, (x_j - x_i)
        """
        if self.directed:
            dx = [((j, i), self.reconstruct_delta_x0(i, j) + int_delta_f[i, j])
                for i in range(self.N) for j in range(self.N) if self.K[i, j] >= self.threshold]
        else:
            dx = [((j, i), self.reconstruct_delta_x0(i, j) + (int_delta_f[i, j] - int_delta_f[j, i])/2)
                for i in range(self.N) for j in range(i+1) if self.K[i, j] >= self.threshold]
            dx += [((i, j), -v) for ((j, i), v) in dx]
        return dx

    def reconstruct_matrix(self):
        """
        Reconstruct matrix K from optimized vector self.C_opt

        Parameters:
        -----------
        None

        Returns:
        --------
        reconstructed matrix, np.ndarray shape(N, N)
        """

        if self.directed:
            K = np.array(
                [np.sqrt((self.C_opt[i, j])**2 + (self.C_opt[i, j+self.N])**2)
                for i in range(self.N) for j in range(self.N)]
            ).reshape(self.N, self.N)
        else:
            Kvals = [
                0.5 * np.sqrt((self.C_opt[i, j] + self.C_opt[j, i])**2 + 
                (self.C_opt[i, j+self.N] - self.C_opt[j, i+self.N])**2)
                for i in range(self.N) for j in range(i)
            ]
            # values of lower triangle, values on diagonal are equal to zero
            K = np.zeros((self.N, self.N))
            upper_ids = np.triu_indices(self.N)     # upper right indices of triangular matrix
            lower_ids = np.tril_indices(self.N, -1) # lower left indices of triangular matrix
            K[lower_ids] = Kvals
            K[upper_ids] = K.T[upper_ids]
        return K

    def reconstruct(self, M=None, method='lstsq'):
        """
        reconstruct the network topology and phase differences, using M measurement steps

        Parameters:
        -----------
        M: int, optional
            number of data points to be used
        s: int, optional
            average over the last s values to obtain delta x at t='infinity'
        method: str
            accepts 'lstsq', 'pinv'
            method used to solve the overdetermined linear system of equations

        Returns:
        -------
        None

        Sets:
        -----
        self.K_raw: the raw reconstructed matrix  
        self.K: the refined reconstructed matrix  
        self.deltax_rec: the reconstructed phase differences for each reconstructed edge  
        self.method: the method used to solve the overdetermined linear system of equations

        """
        self.method = method
        if M is None:  # use all data points for reconstruction
            M = self.M
        self.calculate_Copt(M)

        self.K_rec = self.reconstruct_matrix()
        self.K_raw = self.K_rec * self.dydt_fac
        self.K = deepcopy(self.K_raw)

        self.threshold = self.threshold_func(self.K.ravel())
        self.K[self.K < self.threshold] = 0

        self.deltax_rec = self.reconstruct_delta_x(self.deltax[:, :, :M])  # self.deltax_rec[j][i] = delta_x_ji            

    def get_results(self):
        """
        returns the reconstructed matrix and all reconstructed delta x

        Parameters:
        -----------
        None

        Returns:
        --------
        reconstructed matrix (np.ndarray, shape (N, N)),\n
        reconstructed delta x (list of tuples ((j, i), x_j - x_i))
        """
        return self.K, self.deltax_rec

    def metrics(self, A:np.ndarray=None, x:np.ndarray=None, s:int=100)->dict:
        """
        Compute metrics of reconstruction.

        Parameters:
        -----------
        A: np.ndarray, shape (N, N)  
            Real network, N denotes the number of nodes.   
            If given, Mean Absolute Error, Area under the Receiver-Operator-Characteristics Curve and  
            Average Precicion (Area under Precision-Recall Curve) are calculated.  
        x: np.ndarray, shape (N, M)  
            Real time series of hidden variable x for each node.  
            N denotes the number of nodes, M denotes the number of measurements.
            If given, mean absolute error is calculated, assuming x are phase angles.
        s: int, default s=100
            Assuming absolute convergence of x, s denotes the number of trailing measurements of x
            that are averaged.
        """
        m = dict()
        if A is not None:
            m['mae'] = mean_abs_err_thresholded_data(A, self.K)
            m['auc'] = skl_metrics.roc_auc_score((A > 0).ravel(), self.K_raw.ravel())
            m['ap']  = skl_metrics.average_precision_score((A > 0).ravel(), self.K_raw.ravel())
            m['F1-score'] = skl_metrics.f1_score((A > 0).ravel(), (self.K > 0).ravel())
        if x is not None:
            tmp = np.array([
                [np.mean(x[j, -s:] - x[i, -s:]), dx_rec[-1]] 
                for (j, i), dx_rec in self.deltax_rec])
            # dxreal, dxrec = tmp[:, 0], tmp[:, 1]
            m['mae x'] = mean_abs_err_angles(tmp[:, 0], tmp[:, 1])
        return m



class DirectMethod():
    """ 
    implementation of the 'direct method' to infer network topology from a measured time series,
    adjusted to dynamical systems of the functional form:

    `dot(yi) = g(t, x, y, *args) + sum_j Kij sin(xi - xj)`

    See 
    Timme, Marc, and Jose Casadiego. “Revealing Networks from Dynamics: An Introduction.” Journal of Physics A: Mathematical and Theoretical 47, no. 34 (August 29, 2014): 343001. https://doi.org/10.1088/1751-8113/47/34/343001.
    """
    def __init__(self, t, x, y, *args, g, gkwargs, fac=1, threshold_kwargs={'bins':128}) -> None:
        assert x.shape == y.shape, 'x.shape unequal to y.shape'
        self.N, self.M = x.shape
        self.t = t
        self.x, self.y = x, y
        self.threshold_kwargs = threshold_kwargs

        self.R = np.sin(np.array(
            [self.x[j] - self.x[i] 
            for i in range(self.N) 
            for j in range(self.N)]
        ).reshape(self.N, self.N, self.M))

        self.g_eval = g(t, x, y, *args, **gkwargs)

        if isinstance(fac, np.ndarray) and fac.shape==(self.N,):
            self.fac = fac[:, np.newaxis]
        else:
            self.fac = fac
    
    def calculate_L(self, M=None):
        """ shape: (N, M) """
        if M is None:
            M = self.M
        dot_y = np.gradient(self.y[:, :M], self.t[:M], axis=1)
        self.L = dot_y - self.g_eval[:, :M]
    
    def reconstruct(self, M=None):
        """ shape: (N, N) """
        if not hasattr(self, 'L') or \
            (M is not None and self.L.shape[1] != M) or \
                (self.L.shape[1] != self.M):
            self.calculate_L(M)

        if M and M < self.M:  # use only M data points
            R = self.R[:, :, :M]
        else:  # use all data points
            R = self.R

        self.K_rec = np.array([lstsq(R[i].T, self.L[i].T, rcond=None)[0].T for i in range(self.N)])
        self.K_raw = self.K_rec * self.fac
        self.K = deepcopy(self.K_raw)

        self.threshold = otsu_threshold(self.K.ravel(), **self.threshold_kwargs)
        self.K[self.K < self.threshold] = 0

    def get_results(self):
        """
        returns the reconstructed matrix

        Parameters:
        -----------
        None

        Returns:
        --------
        reconstructed matrix (np.ndarray, shape (N, N)),\n
        """
        return self.K

    def metrics(self, A:np.ndarray=None)->dict:
        """
        Compute metrics of reconstruction.

        Parameters:
        -----------
        A: np.ndarray, shape (N, N)  
            Real network, N denotes the number of nodes.   
            If given, Mean Absolute Error, Area under the Receiver-Operator-Characteristics Curve and  
            Average Precicion (Area under Precision-Recall Curve) are calculated.  
        """
        m = dict()
        if A is not None:
            m['mae'] = mean_abs_err_thresholded_data(A, self.K)
            m['auc'] = skl_metrics.roc_auc_score((A > 0).ravel(), self.K_raw.ravel())
            m['ap']  = skl_metrics.average_precision_score((A > 0).ravel(), self.K_raw.ravel())
            m['F1-score'] = skl_metrics.f1_score((A > 0).ravel(), (self.K > 0).ravel())
        return m

