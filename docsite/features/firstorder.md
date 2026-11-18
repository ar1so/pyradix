# First order

First-order features describe the distribution of voxel intensities inside the region, without regard to where the
voxels are. Most use the raw intensities \(\mathbf{X}\); **Entropy** and **Uniformity** use the discretised histogram
\(p(i)\). Notation: see [Feature reference](index.md#notation).

| Feature | Definition |
|---|---|
| **Energy** | \(\displaystyle \sum_{i=1}^{N_p} \big(\mathbf{X}(i) + c\big)^2\) |
| **TotalEnergy** | \(\displaystyle V_{voxel} \sum_{i=1}^{N_p} \big(\mathbf{X}(i) + c\big)^2\) &nbsp; (not in IBSI) |
| **Entropy** | \(\displaystyle -\sum_{i=1}^{N_g} p(i) \log_2\big(p(i)+\epsilon\big)\) &nbsp; (IBSI: intensity-histogram entropy) |
| **Minimum** | \(\min(\mathbf{X})\) |
| **10Percentile** | 10th percentile of \(\mathbf{X}\) |
| **90Percentile** | 90th percentile of \(\mathbf{X}\) |
| **Maximum** | \(\max(\mathbf{X})\) |
| **Mean** | \(\displaystyle \bar{X} = \frac{1}{N_p}\sum_{i=1}^{N_p} \mathbf{X}(i)\) |
| **Median** | median of \(\mathbf{X}\) |
| **InterquartileRange** | \(P_{75} - P_{25}\) |
| **Range** | \(\max(\mathbf{X}) - \min(\mathbf{X})\) |
| **MeanAbsoluteDeviation** | \(\displaystyle \frac{1}{N_p}\sum_{i=1}^{N_p} \lvert \mathbf{X}(i) - \bar{X} \rvert\) |
| **RobustMeanAbsoluteDeviation** | Mean absolute deviation of the intensities between the 10th and 90th percentile, from their own mean |
| **RootMeanSquared** | \(\displaystyle \sqrt{\frac{1}{N_p}\sum_{i=1}^{N_p} \big(\mathbf{X}(i)+c\big)^2}\) |
| **StandardDeviation** *(deprecated)* | \(\sqrt{\text{Variance}}\); duplicates Variance, not in IBSI |
| **Skewness** | \(\displaystyle \frac{\mu_3}{\sigma^3}\), with \(\mu_k = \frac{1}{N_p}\sum (\mathbf{X}(i)-\bar{X})^k\) and \(\sigma^2 = \mu_2\); 0 for a flat region |
| **Kurtosis** | \(\displaystyle \frac{\mu_4}{\sigma^4}\) in `pyradiomics` mode; \(\displaystyle \frac{\mu_4}{\sigma^4} - 3\) (excess kurtosis) in `ibsi` mode |
| **Variance** | \(\displaystyle \sigma^2 = \frac{1}{N_p}\sum_{i=1}^{N_p} (\mathbf{X}(i) - \bar{X})^2\) |
| **Uniformity** | \(\displaystyle \sum_{i=1}^{N_g} p(i)^2\) &nbsp; (IBSI: intensity-histogram uniformity) |

!!! note "Kurtosis is the one first-order feature whose formula differs between the modes"
    IBSI defines excess kurtosis (a normal distribution gives 0); PyRadiomics reports the uncorrected value (a normal
    distribution gives 3). The two differ by exactly 3.

!!! tip "`voxelArrayShift`"
    PyRadiomics can add a constant \(c\) before Energy, TotalEnergy and RootMeanSquared so that negative intensities
    do not cancel out. Its default is 0, and its documentation suggests a value such as 2000 for CT. IBSI adds nothing.
    pyradix also defaults to \(c = 0\) in both modes; set `voxelArrayShift` to the value an older PyRadiomics run used.
