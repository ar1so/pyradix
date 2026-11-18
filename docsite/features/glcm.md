# GLCM: grey-level co-occurrence matrix

The GLCM counts how often two grey levels occur next to each other. For a distance \(\delta\) and direction
\(\theta\), \(\mathbf{P}(i,j)\) is the number of times a voxel with grey level \(i\) has a neighbour with grey level
\(j\) along \(\theta\). pyradix uses all 13 directions in 3D (4 in 2D), distance 1 by default (`distances`), and a
symmetrical matrix (`symmetricalGLCM`). Features are computed per direction and averaged, unless `weightingNorm`
merges the directions first.

Notation: \(p(i,j) = \mathbf{P}(i,j) / \sum \mathbf{P}\); marginals \(p_x(i) = \sum_j p(i,j)\) and
\(p_y(j) = \sum_i p(i,j)\); means \(\mu_x, \mu_y\) and standard deviations \(\sigma_x, \sigma_y\) of \(p_x, p_y\);
\(p_{x+y}(k) = \sum_{i+j=k} p(i,j)\) for \(k = 2 \dots 2N_g\) and \(p_{x-y}(k) = \sum_{|i-j|=k} p(i,j)\) for
\(k = 0 \dots N_g-1\). Entropies: \(HX = -\sum p_x \log_2(p_x+\epsilon)\), \(HY\) likewise,
\(HXY = -\sum p(i,j)\log_2(p(i,j)+\epsilon)\),
\(HXY1 = -\sum p(i,j)\log_2(p_x(i)p_y(j)+\epsilon)\),
\(HXY2 = -\sum p_x(i)p_y(j)\log_2(p_x(i)p_y(j)+\epsilon)\).

| Feature | Definition |
|---|---|
| **Autocorrelation** | \(\sum_{i,j} p(i,j)\, i\, j\) |
| **JointAverage** | \(\mu_x = \sum_{i,j} p(i,j)\, i\) |
| **ClusterProminence** | \(\sum_{i,j} (i + j - \mu_x - \mu_y)^4\, p(i,j)\) |
| **ClusterShade** | \(\sum_{i,j} (i + j - \mu_x - \mu_y)^3\, p(i,j)\) |
| **ClusterTendency** | \(\sum_{i,j} (i + j - \mu_x - \mu_y)^2\, p(i,j)\) |
| **Contrast** | \(\sum_{i,j} (i-j)^2\, p(i,j)\) |
| **Correlation** | \(\dfrac{\sum_{i,j} p(i,j)\, i\, j - \mu_x \mu_y}{\sigma_x \sigma_y}\); 1 for a flat region |
| **DifferenceAverage** | \(\sum_{k} k\, p_{x-y}(k)\) |
| **DifferenceEntropy** | \(-\sum_{k} p_{x-y}(k) \log_2\big(p_{x-y}(k)+\epsilon\big)\) |
| **DifferenceVariance** | \(\sum_{k} (k - \text{DifferenceAverage})^2\, p_{x-y}(k)\) |
| **JointEnergy** | \(\sum_{i,j} p(i,j)^2\) |
| **JointEntropy** | \(HXY\) |
| **Imc1** | \(\dfrac{HXY - HXY1}{\max(HX, HY)}\) |
| **Imc2** | \(\sqrt{1 - e^{-2(HXY2 - HXY)}}\) |
| **Idm** | \(\sum_{i,j} \dfrac{p(i,j)}{1 + \lvert i-j \rvert^2}\) (inverse difference moment) |
| **Idmn** | \(\sum_{i,j} \dfrac{p(i,j)}{1 + \lvert i-j \rvert^2 / N_g^2}\) (normalised) |
| **Id** | \(\sum_{i,j} \dfrac{p(i,j)}{1 + \lvert i-j \rvert}\) (inverse difference) |
| **Idn** | \(\sum_{i,j} \dfrac{p(i,j)}{1 + \lvert i-j \rvert / N_g}\) (normalised) |
| **InverseVariance** | \(\sum_{k \ge 1} \dfrac{p_{x-y}(k)}{k^2}\) |
| **MaximumProbability** | \(\max_{i,j} p(i,j)\) |
| **MCC** | \(\sqrt{\text{second largest eigenvalue of } Q}\), with \(Q(i,j) = \sum_k \dfrac{p(i,k)\,p(j,k)}{p_x(i)\,p_y(k)}\) (maximal correlation coefficient) |
| **SumAverage** | \(\sum_{k} k\, p_{x+y}(k)\); equals \(2\,\mu_x\) for a symmetrical GLCM |
| **SumEntropy** | \(-\sum_{k} p_{x+y}(k) \log_2\big(p_{x+y}(k)+\epsilon\big)\) |
| **SumSquares** | \(\sum_{i,j} (i - \mu_x)^2\, p(i,j)\) |
| **Dissimilarity** *(deprecated)* | \(\sum_{i,j} \lvert i-j \rvert\, p(i,j)\); equals DifferenceAverage |
| **Homogeneity1** *(deprecated)* | equals Id |
| **Homogeneity2** *(deprecated)* | equals Idm |
| **SumVariance** *(deprecated)* | \(\sum_k (k - \text{SumAverage})^2\, p_{x+y}(k)\); equals ClusterTendency |

!!! note "MCC and LAPACK"
    MCC needs the eigenvalues of a non-symmetric matrix. pyradix computes them with LAPACK `dgeev`, the routine NumPy
    and PyRadiomics use (Accelerate on macOS, OpenBLAS on Linux), and with the pure-Rust `faer` library on Windows.
