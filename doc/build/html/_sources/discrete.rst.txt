FEM for discrete data
=====================

Here, we describe the version of FEM that requires discrete data, that is variables :math:`x_i,y` which take on values from a finite set of symbols. In biology, such data may occur naturally (the DNA sequences that form genes or the amino acid sequences that form proteins, for example) or may result from discretizing continuous variables (assigning neurons' states to on or off, for example).

Model
-----

The distribution :math:`p` that we wish to learn operates on the *one-hot* encodings of discrete variables defined as follows. Assume the variable :math:`x_i` takes on one of :math:`m_i` states symbolized by the first :math:`m_i` positive integers, i.e. :math:`x_i\in\{1,2,\ldots,m_i\}`. The one-hot encoding :math:`\sigma_i\in\{0,1\}^{m_i}` of :math:`x_i` is a vector of length :math:`m_i` whose :math:`j^{th}`, :math:`j=1,\ldots,m_i` component is

.. math::

   \sigma_{ij}(x_i) = \begin{cases} 1 & \text{ if }x_i=j \\ 0 & \text{otherwise}\end{cases}

Note that :math:`\sigma_i` is a boolean vector with exactly one 1 and the rest 0's. Assume that we observe :math:`n` variables, then the state of the input is represented by the vector :math:`\sigma=\sum_{i=1}^ne_i\otimes\sigma_i` where :math:`e_i` is the :math:`i^{th}` canonical basis vector of :math:`\mathbb{Z}^n`. In other words, :math:`\sigma=\begin{pmatrix}\sigma_1&\cdots&\sigma_n\end{pmatrix}^T\in\{0,1\}^{M}` where :math:`M=\sum_{j=1}^nm_j` is formed from concatenating the one-hot encodings of each input variable. Let :math:`\mathcal{S}` denote the set of valid :math:`\sigma`.

Assume the output variable :math:`y` takes on one of :math:`m` values, i.e. :math:`y\in\{1,\ldots,m\}`, then the probability distribution :math:`p:\mathcal{S}\rightarrow [0,1]` is defined by

.. math::

   p(y=j~|~\sigma) = {e^{h_j(\sigma)} \over \sum_{i=1}^{m} e^{h_i(\sigma)}}

where :math:`h_i(\sigma)` is the negative energy of the :math:`i^{th}` state of :math:`y` when the input state is :math:`\sigma`. :math:`p(y=j~|~\sigma)` is the probability according to the `Boltzmann distribution`_ that :math:`y` is in state :math:`j` given that the input is in the state represented by :math:`\sigma`.

Importantly, :math:`h:\mathcal{S}\rightarrow\mathbb{R}^m` maps :math:`\sigma` to the negative energies of states of :math:`y` in an interpretable manner:

.. math::

    h(\sigma) = \sum_{k=1}^KW_k\sigma^k.

The primary objective of FEM is to determine the model parameters that make up the matrices :math:`W_k`. :math:`\sigma^k` is a vector of distinct powers of :math:`\sigma` components.

The shapes of :math:`W_k` and :math:`\sigma^k` are :math:`m\times p_k` and :math:`p_k\times1`, respectively, where :math:`p_k=\sum_{A\subseteq\{1,\ldots,n\}, |A|=k}\prod_{j\in A}m_j`. The number of terms in the sum defining :math:`p_k` is :math:`{n \choose k}`, the number of ways of choosing :math:`k` out of the :math:`n` input variables. The products in the formula for :math:`p_k` reflect the fact that input variable :math:`x_j` can take :math:`m_j` states. Note that if all :math:`m_j=m`, then :math:`p_k={n\choose k}m^k`, the number ways of choosing :math:`k` input variables each of which may be in one of :math:`m` states.

For example, if :math:`n=2` and :math:`m_1=m_2=3`, then

.. math::

   \sigma^1 = \begin{pmatrix} \sigma_{11} & \sigma_{12} & \sigma_{13} & \sigma_{21} & \sigma_{22} & \sigma_{23} \end{pmatrix}^T,

which agrees with the definition of :math:`\sigma` above, and

.. math::
   
   \sigma^2 = \begin{pmatrix} \sigma_{11}\sigma_{21} & \sigma_{11}\sigma_{22} & \sigma_{11}\sigma_{23} & \sigma_{12}\sigma_{21} & \sigma_{12}\sigma_{22} & \sigma_{12}\sigma_{23} & \sigma_{13}\sigma_{21} & \sigma_{13}\sigma_{22} & \sigma_{13}\sigma_{23} \end{pmatrix}^T.

Note that we exclude powers of the form :math:`\sigma_{ij_1}\sigma_{ij_2}` with :math:`j_1\neq j_2` since they are guaranteed to be 0. On the other hand, we exclude powers of the form :math:`\sigma_{ij}^k` for :math:`k>1` since they are guaranteed to be 1 as long as :math:`\sigma_{ij}=1` and therefore would be redundant to the linear terms in :math:`h.` For those reasons, :math:`\sigma^k` for :math:`k>2` is empty in the above example, and generally the greatest degree of :math:`h` must satisfy :math:`K\leq n`, though this is hardly as restrictive as are computing abilities in real applications.

We say that :math:`h` is interpretable because the effect of interactions between the input variables on the output variable is evident from the parameters :math:`W_k`. Consider the explicit formula for :math:`h` for the example above with :math:`m=2`:

.. math::

   \begin{pmatrix} h_1(\sigma) \\ h_2(\sigma) \end{pmatrix} = \underbrace{\begin{pmatrix} W_{111} & W_{112} \\ W_{121} & W_{122} \end{pmatrix}}_{W_1} \begin{pmatrix} \sigma_{11} \\ \sigma_{12} \\ \sigma_{13} \\ \sigma_{21} \\ \sigma_{22} \\ \sigma_{23}\end{pmatrix} + \underbrace{\begin{pmatrix} W_{21(1,2)} \\ W_{22(1,2)} \end{pmatrix}}_{W_2}\begin{pmatrix} \sigma_{11}\sigma_{21} \\ \sigma_{11}\sigma_{22} \\ \sigma_{11}\sigma_{23} \\ \sigma_{12}\sigma_{21} \\ \sigma_{12}\sigma_{22} \\ \sigma_{12}\sigma_{23} \\ \sigma_{13}\sigma_{21} \\ \sigma_{13}\sigma_{22} \\ \sigma_{13}\sigma_{23} \end{pmatrix}.

We've written :math:`W_1` as a block matrix with :math:`1\times m_j` row vector blocks :math:`W_{1ij}=\begin{pmatrix}W_{1ij1}&\cdots&W_{1ijm_j}\end{pmatrix}` that describe the effect of :math:`x_j` on :math:`y_i`. In particular, recalling that the probability of :math:`y=i` given a input state :math:`\sigma` is the :math:`i^{th}` component of

.. math::
   
   p(y~|~\sigma) = {1 \over e^{h_1(\sigma)}+e^{h_2(\sigma)}} \begin{pmatrix} e^{h_1(\sigma)} \\ e^{h_2(\sigma)} \end{pmatrix}

we see that :math:`h_i(\sigma)` and hence the probability of :math:`y=i` increases as :math:`W_{1ijs}` increases when :math:`x_j=s`. In general, :math:`W_k` can be written as :math:`n` rows each with :math:`{n \choose k}` blocks :math:`W_{ki\lambda}` of shape :math:`1\times\prod_{j\in\lambda}m_j` where :math:`\lambda=(j_1,\ldots,j_k)`, which represent the effect that variables :math:`x_{j_1},\ldots,x_{j_k}` collectively have on :math:`y_i`. That is :math:`h_i(\sigma)` and hence the probability of :math:`y=i` increases as :math:`W_{k1\lambda s}` increases when :math:`x_{j_1}=s_1,\ldots,x_{j_k}=s_k`, where :math:`\lambda=(j_1,\ldots,j_k)` and :math:`s=(s_1,\ldots,s_k)`.

.. plot:: scripts/w.py


Method
------

Suppose we make :math:`\ell` observations of the variables :math:`x_i, y`. We may arrange the one-hot encodings of these observations into matrices. Let :math:`\Sigma_{xk}`, :math:`k=1,\ldots,K`, be the matrix whose :math:`j^{th}` column is the :math:`k^{th}` power of the one-hot encoding of the :math:`j^{th}` input observation :math:`\sigma_j^k`. Similarly, let :math:`\Sigma_y` be the matrix whose :math:`j^{th}` column is the one-hot encoding of the :math:`j^{th}` output observation.

We summarize the probability of :math:`y=i` given input observation :math:`\sigma_j` in the matrix :math:`P(\Sigma_y~|~W)` with elements

.. math::

   P_{ij} = {e^{H_{ij}} \over \sum_{i=1}^m e^{H_{ij}}},


where :math:`H_{ij}` are the elements of the the matrix :math:`H = W\Sigma_x` with

.. math::

   W = \begin{pmatrix} W_1 & \cdots & W_K \end{pmatrix}\hspace{5mm}\text{and}\hspace{5mm}\Sigma_x = \begin{pmatrix} \Sigma_{x1} \\ \vdots \\ \Sigma_{xK} \end{pmatrix}.

:math:`\Sigma_x` and :math:`\Sigma_y` are computed solely from the data. We can adjust a guess at :math:`W` by comparing the corresponding :math:`H` and :math:`P(\Sigma_y~|~W)`, computed using the formulas above, to :math:`\Sigma_y`. That is, after modifying :math:`H` to reduce the difference :math:`\Sigma_y-P(\Sigma_y~|~W)` we can solve the formula :math:`H=W\Sigma_x` for the model parameters :math:`W`. This is the motivation for the following method:

   Initialize :math:`W^{(1)}=0` 

   Repeat for :math:`k=1,2,\ldots` until convergence:

      :math:`H^{(k)} = W^{(k)}\Sigma_x`

      :math:`P_{ij}^{(k)} = {e^{H^{(k)}_{ij}} \over \sum_{i=1}^m e^{H^{(k)}_{ij}}}`

      :math:`H^{(k+1)} = H^{(k)}+\Sigma_y-P^{(k)}`

      Solve :math:`W^{(k+1)}\Sigma_x = H^{(k+1)}` for :math:`W^{(k+1)}`

The shapes of all matrices mentioned in this section are listed in the following table:

+-------------------------+-----------------------------------+
| matrix                  | shape                             |
+=========================+===================================+
| :math:`\Sigma_x`        | :math:`\sum_{k=1}^np_k\times\ell` |
+-------------------------+-----------------------------------+
| :math:`\Sigma_{xk}`     | :math:`p_k\times\ell`             |
+-------------------------+-----------------------------------+
| :math:`\Sigma_y`        | :math:`m\times\ell`               |
+-------------------------+-----------------------------------+
| :math:`P(\Sigma_y~|~W)` | :math:`m\times\ell`               |
+-------------------------+-----------------------------------+
| :math:`H`               | :math:`m\times\ell`               |
+-------------------------+-----------------------------------+
| :math:`W`               | :math:`m\times\sum_{k=1}^np_k`    |
+-------------------------+-----------------------------------+
| :math:`W_k`             | :math:`m\times p_k`               |
+-------------------------+-----------------------------------+

where :math:`p_k=\sum_{A\subseteq\{1,\ldots,n\}, |A|=k}\prod_{j\in A}m_j`.

.. _Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
