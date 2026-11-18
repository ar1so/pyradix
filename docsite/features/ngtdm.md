# NGTDM: neighbouring grey-tone difference matrix

For each grey level \(i\), the NGTDM sums how much voxels of that grey level differ from the average of their
neighbourhood (distance \(\delta\), default 1):

\[
s_i = \sum_{\text{voxels with grey level } i} \big\lvert i - \bar{A}_i \big\rvert
\]

where \(\bar{A}_i\) is the average grey level of the voxel's neighbours inside the region. Let \(n_i\) be the
number of such voxels, \(N_{v,p} = \sum n_i\) the number of voxels with a valid neighbourhood, \(p_i = n_i / N_{v,p}\),
and \(N_{g,p}\) the number of grey levels with \(p_i \neq 0\).

| Feature | Definition |
|---|---|
| **Coarseness** | \(\dfrac{1}{\sum_i p_i s_i}\); capped at \(10^6\) when the denominator is 0 |
| **Contrast** | \(\Big(\dfrac{1}{N_{g,p}(N_{g,p}-1)}\sum_{i,j} p_i p_j (i-j)^2\Big)\Big(\dfrac{1}{N_{v,p}}\sum_i s_i\Big)\), over \(p_i, p_j \neq 0\) |
| **Busyness** | \(\dfrac{\sum_i p_i s_i}{\sum_{i,j} \lvert i p_i - j p_j \rvert}\), over \(p_i, p_j \neq 0\) |
| **Complexity** | \(\dfrac{1}{N_{v,p}}\sum_{i,j} \lvert i-j \rvert \dfrac{p_i s_i + p_j s_j}{p_i + p_j}\), over \(p_i, p_j \neq 0\) |
| **Strength** | \(\dfrac{\sum_{i,j} (p_i + p_j)(i-j)^2}{\sum_i s_i}\), over \(p_i, p_j \neq 0\) |

!!! note "Difference between the modes"
    IBSI counts a voxel only when at least one neighbour lies inside the region. PyRadiomics 3.1.0's documentation
    states the same rule, but its C implementation also counts voxels without any neighbour in the region (adding a
    difference of 0). `pyradiomics` mode reproduces the implementation, so that values match PyRadiomics exactly;
    `ibsi` mode follows the IBSI rule.
