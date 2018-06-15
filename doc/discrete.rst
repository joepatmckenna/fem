FEM for discrete data
=====================

Here, we describe the version of FEM that requires discrete data, that is variables :math:`x_i,y` which take on values from a finite set of symbols. In biology, such data may occur naturally (the DNA sequences that form genes or the amino acid sequences that form proteins, for example) or may result from discretizing continuous variables (assigning neurons' states to on or off, for example).

Model
-----

The function :math:`f` that we wish to learn operates on the "one-hot" encodings of discrete variables defined as follows. Assume the variable :math:`x_i` takes on one of :math:`m_i` states symbolized by the first :math:`m_i` positive integers, i.e. :math:`x_i\in\{1,2,\ldots,m_i\}`. The one-hot encoding :math:`\sigma_i\in\{0,1\}^{m_i}` of :math:`x_i` is a vector of length :math:`m_i` whose :math:`j^{th}`, :math:`j=1,\ldots,m_i` component is

.. math::

   \sigma_{ij}(x_i) = \begin{cases} 1 & \text{ if }x_i=j \\ 0 & \text{otherwise}\end{cases}

Note that :math:`\sigma_i` is a boolean vector with exactly one 1 and the rest 0's. Assume that we observe :math:`n` variables, then the state of the system is represented by the vector :math:`\sigma=\begin{pmatrix}\sigma_1&\cdots&\sigma_n\end{pmatrix}^T` formed from concatenating the one-hot encodings of each input variable. The set of valid :math:`\sigma` is :math:`\mathcal{S} = \{\sigma\in\{0,1\}^{M_{n+1}}:\sum_{j=M_i+1}^{M_{i+1}}\sigma_{ij}=1\text{ for each }i=1,\ldots,n\}` with :math:`M_i=\sum_{j<i}m_j`.

Assume the output variable :math:`y` takes on one of :math:`m` values, i.e. :math:`y\in\{1,\ldots,m\}`, then :math:`f:\mathcal{S}\rightarrow [0,1]^m` is defined as

.. math::

   f(\sigma) = {1 \over \sum_{i=1}^{m} e^{h_i(\sigma)}} \begin{pmatrix} e^{h_1(\sigma)} \cdots e^{h_m(\sigma)} \end{pmatrix}^T

where :math:`h_i(\sigma)` is the negative energy of the :math:`i^{th}` state of :math:`y` when the system is in the state :math:`\sigma`. The :math:`i^{th}` component of :math:`f(\sigma)` is the probability according to the `Boltzmann distribution`_ that :math:`y` is in state :math:`i` given that the system is in the state :math:`\sigma`.

Importantly, :math:`h:\mathcal{S}\rightarrow\mathbb{R}^m` maps :math:`\sigma` to the negative energies of states of :math:`y` in an interpretable manner:

.. math::

    h(\sigma) = \sum_{k=1}^KH^k\sigma^k.

The primary objective of FEM is to determine the model parameters that make up the matrices :math:`H^k`. :math:`\sigma^k` is a vector of distinct powers of :math:`\sigma` components computed from the data.

The shapes of :math:`H^k` and :math:`\sigma_k` are :math:`m\times p_k` and :math:`p_k\times1`, respectively, where :math:`p_k=\sum_{S\subseteq\{1,\ldots,n\}}\prod_{j\in S}m_j`. The number of terms in the sum defining :math:`p_k` is :math:`{n \choose k}`, the number of ways of choosing :math:`k` out of the :math:`n` input variables. The products in the formula for :math:`p_k` reflect the fact that input variable :math:`x_j` can take :math:`m_j` states. Note that if all :math:`m_j=m`, then :math:`p_k={n\choose k}m^k`, the number ways of choosing :math:`k` input variables each of which may be in one of :math:`m` states.

For example, if :math:`n=2` and :math:`m_1=m_2=3`, then

.. math::

   \sigma^1 = \begin{pmatrix} \sigma_{11} & \sigma_{12} & \sigma_{13} & \sigma_{21} & \sigma_{22} & \sigma_{23} \end{pmatrix}^T,

which agrees with the definition of :math:`\sigma` above, and

.. math::
   
   \sigma^2 = \begin{pmatrix} \sigma_{11}\sigma_{21} & \sigma_{11}\sigma_{22} & \sigma_{11}\sigma_{23} & \sigma_{12}\sigma_{21} & \sigma_{12}\sigma_{22} & \sigma_{12}\sigma_{23} & \sigma_{13}\sigma_{21} & \sigma_{13}\sigma_{22} & \sigma_{13}\sigma_{23} \end{pmatrix}^T.

Note that we exclude powers of the form :math:`\sigma_{ij_1}\sigma_{ij_2}` with :math:`j_1\neq j_2` since they are guaranteed to be 0. On the other hand, we exclude powers of the form :math:`\sigma_{ij}^k` for :math:`k>1` since they are guaranteed to be 1 as long as :math:`\sigma_{ij}=1` and therefore would be redundant to the linear terms in :math:`h.` For those reasons, :math:`\sigma^k` for :math:`k>2` is empty in the above example, and generally the greatest degree of :math:`h` must satisfy :math:`K\leq n`.

We say that :math:`h` is interpretable because the effect of interactions between the input variables on the output variable is evident from the parameters :math:`H^k`. Consider the explicit formula for :math:`h` for the example above with :math:`m=2`:

.. math::

   \begin{pmatrix} h_1(\sigma) \\ h_2(\sigma) \end{pmatrix} = \underbrace{\begin{pmatrix} H^1_{11} & H^1_{12} \\ H^1_{21} & H^1_{22} \end{pmatrix}}_{H^1} \begin{pmatrix} \sigma_{11} \\ \sigma_{12} \\ \sigma_{13} \\ \sigma_{21} \\ \sigma_{22} \\ \sigma_{23}\end{pmatrix} + \underbrace{\begin{pmatrix} H^2_{1,(1,2)} \\ H^2_{2,(1,2)} \end{pmatrix}}_{H^2}\begin{pmatrix} \sigma_{11}\sigma_{21} \\ \sigma_{11}\sigma_{22} \\ \sigma_{11}\sigma_{23} \\ \sigma_{12}\sigma_{21} \\ \sigma_{12}\sigma_{22} \\ \sigma_{12}\sigma_{23} \\ \sigma_{13}\sigma_{21} \\ \sigma_{13}\sigma_{22} \\ \sigma_{13}\sigma_{23} \end{pmatrix}.

We've written :math:`H^1` as a block matrix with :math:`1\times m_j` row vector blocks :math:`H^1_{ij}=\begin{pmatrix}H^{11}_{ij}&\cdots&H^{1m_j}_{ij}\end{pmatrix}` that describe the effect of :math:`x_j` on :math:`y_i`. In particular, recalling that the probability of :math:`y=i` given a system state :math:`\sigma` is the :math:`i^{th}` component of

.. math::
   
   f(\sigma) = {1 \over e^{h_1(\sigma)}+e^{h_2(\sigma)}} \begin{pmatrix} e^{h_1(\sigma)} \\ e^{h_2(\sigma)} \end{pmatrix}

we see that if :math:`x_j=s`, then :math:`h_i(\sigma)` and hence the probability of :math:`y=i` increases as :math:`H^{1s}_{ij}` increases. In general, :math:`H^k` can be written as :math:`n` rows each with :math:`{n \choose k}` blocks :math:`H^k_{i\lambda}` of shape :math:`1\times\prod_{j\in\lambda}m_j` where :math:`\lambda=(j_1,\ldots,j_k)`, which represent the effect that variables :math:`x_{j_1},\ldots,x_{j_k}` collectively have on :math:`y_i`.

.. plot:: _scripts/h.py

Method
------

Suppose that we observe the variables :math:`x_i, y` many times, say :math:`\ell` times. We may arrange the one-hot encoding of each observation and their powers into matrices. Let :math:`\Sigma^1` be the matrix whose :math:`j^{th}` column :math:`\sigma_j` is the one-hot encoding of the :math:`j^{th}` observation of the input :math:`x_i`, and let :math:`\Sigma^k` be the matrix whose :math:`j^{th}` column is the :math:`k^{th}` power of :math:`\sigma_j`, :math:`\sigma_j^k`. We may then summarize the probability of :math:`y=i` given observation :math:`\sigma_j` as entry

.. math::

   F_{ij} = {1 \over \sum_{i=1}^m e^{h_{ij}}} \begin{pmatrix} e^{h_{1j}} & \cdots & e^{h_{mj}} \end{pmatrix}^T


of the matrix :math:`F`. Here the negative energy of the :math:`i^{th}` state of :math:`y` given observation :math:`\sigma_j` is the :math:`ij^{th}` element of the matrix :math:`h = H\Sigma` where

.. math::

   H = \begin{pmatrix} H^1 & \cdots & H^K \end{pmatrix}\hspace{5mm}\text{and}\hspace{5mm}\Sigma = \begin{pmatrix} \Sigma^1 \\ \vdots \\ \Sigma^K \end{pmatrix}

The method is

   initialize :math:`W^{(1)}=0` then :math:`H^{(1)} = W^{(1)}\Sigma`

   repeat for :math:`k=1,2,\ldots` until convergence:

      :math:`P_{ij} = {e^{H^{(k)}_{ij}} \over \sum_{i=1}^m e^{H^{(k)}_{ij}}}`

      :math:`H^{(k+1)} = H^{(k)}+\Sigma_y-P`

      :math:`W^{(k+1)} = H^{(k+1)}VS^+U^T`

+-------------------+-------------------------+
| matrix            | shape                   |
+===================+=========================+
| :math:`F`         | :math:`m\times \ell`    |
+-------------------+-------------------------+
| :math:`h`         | :math:`m\times \ell`    |
+-------------------+-------------------------+
| :math:`H^k`       | :math:`m\times p_k`     |
+-------------------+-------------------------+
| :math:`H`         | :math:`m\times p`       |
+-------------------+-------------------------+
| :math:`\Sigma^k`  | :math:`p_k\times \ell`  |
+-------------------+-------------------------+
| :math:`\Sigma`    | :math:`p\times \ell`    |
+-------------------+-------------------------+
| :math:`\Sigma_y`  | :math:`m\times \ell`    |
+-------------------+-------------------------+
| :math:`U`         | :math:`p\times r`       |
+-------------------+-------------------------+
| :math:`S`         | :math:`r\times r`       |
+-------------------+-------------------------+
| :math:`V`         | :math:`r\times r`       |
+-------------------+-------------------------+


.. _Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
