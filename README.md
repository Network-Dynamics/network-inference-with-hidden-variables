# Network inference with inaccessible variables
### Overview
Code to  reconstruct the network topology in dynamical systems with following the functional form  

$\dfrac{dx_i}{dt} = f(\mathbf{y}, t)$  
  
$\dfrac{dy_i}{dt} = g(\mathbf{y}, t) + \sum_{j=1}^{n} \sin{\left(x_j(t) - x_i(t)\right)}$

assuming $f,g$ and $y_i$ are known for all $i=0,\ldots N$.  

See *R.Schmidt et al., Inferring topology of networks with hidden dynamic variables, IEEE Access (2022)* for the publication.

### Example
Let $y\in\mathbb{R}^{N\times M}$ be the time series recorded at time points $t\in\mathbb{R}^M$ of all $N$ nodes in the network and let the dynamical system be given by the equations above. To reconstruct the network topology, use  

`from reconstruction import Reconstructor`  
`R = Reconstructor(y, t, f, g, fkwargs, gkwargs)`  
`R.reconstruct()`  

Here, `f=f(t, y, **fkwargs)` and `g=g(t, y, **gkwargs)` are callables.  

Results can be accessed via  

`K, delta_x = R.get_results()`

where `K` denotes the reconstructed matrix, and `delta_x` is a list of tuples. Each tuple contains `(j, i)` and the reconstructed `delta_x_ij`, i.e. the reconstructed difference of $x_j - x_i$ at the edge $j\rightarrow i$.


### Requirements
The code requires the following packages:  
`numpy, scipy, numba, sklearn`

The code has been tested using  
`python   3.8.5`  
`numpy    1.19.3`  
`scipy    1.6.0`  
`numba    0.50.1`  
`sklearn  0.23.2`

### Numerical outliers
Reconstruction is implented using either least squares (`np.linalg.lstsq`) or pseudo inverse (`np.linalg.pinv`), which can be specified via `R.reconstruct(method='lstsq')`. When using `method='pinv'`, rarely, numerical outliers might occur when inferring the network topology, see Fig. 1(a).

![](/images/outlier.png)
**Fig.1** *Outliers in inferred coupling matrix caused by single-value computation.*

This is due to the calculation of the pseudo inverse via singular value decomposition. In the calculation, 1 is divided by the individual singular values unequal to zero. If a singular value is very small however, single very large matrix elements may occur in the Moore-Penrose-Pseudoinverse $R^{i+}$.  
Typically, we separate numerical noise from the reconstructed edges by calculating a threshold value. However, if there is a single, very large matrix element, the threshold value also becomes very large, which in turn is responsible for omitting all inferred edges except for the outlier, see Fig. 1(b). As only a single, very large matrix element remains, the computed MAE also is very large. Since detecting this computational error is very easy, e.g. by comparing the number of edges the reconstructed matrix with the (known) number of nodes, we discarded the corresponding simulation in Fig. 3 in the publication.
