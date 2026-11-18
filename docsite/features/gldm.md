# GLDM: grey-level dependence matrix

For each voxel, the **dependence** is the number of neighbours (within distance \(\delta\), default 1) whose grey
level differs from the voxel's by at most \(\alpha\) (`gldm_a`, default 0), plus one for the voxel itself.
\(\mathbf{P}(i,j)\) is the number of voxels with grey level \(i\) and dependence \(j\).

Notation: \(N_z = \sum_{i,j}\mathbf{P}(i,j)\) (equal to the number of voxels \(N_p\)),
\(p(i,j) = \mathbf{P}(i,j)/N_z\), and \(\mu_i = \sum_{i,j} p(i,j)\,i\), \(\mu_j = \sum_{i,j} p(i,j)\,j\).

| Feature | Definition |
|---|---|
| **SmallDependenceEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{j^2}\) |
| **LargeDependenceEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, j^2\) |
| **GrayLevelNonUniformity** | \(\dfrac{1}{N_z}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **GrayLevelNonUniformityNormalized** | \(\dfrac{1}{N_z^2}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **DependenceNonUniformity** | \(\dfrac{1}{N_z}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **DependenceNonUniformityNormalized** | \(\dfrac{1}{N_z^2}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **GrayLevelVariance** | \(\sum_{i,j} p(i,j)\,(i - \mu_i)^2\) |
| **DependenceVariance** | \(\sum_{i,j} p(i,j)\,(j - \mu_j)^2\) |
| **DependenceEntropy** | \(-\sum_{i,j} p(i,j) \log_2\big(p(i,j)+\epsilon\big)\) |
| **LowGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2}\) |
| **HighGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, i^2\) |
| **SmallDependenceLowGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2 j^2}\) |
| **SmallDependenceHighGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, i^2}{j^2}\) |
| **LargeDependenceLowGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, j^2}{i^2}\) |
| **LargeDependenceHighGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, i^2 j^2\) |
| **DependencePercentage** *(deprecated)* | \(N_z / N_p\); always 1, because every voxel has a dependence |

IBSI calls this matrix the neighbouring grey-level dependence matrix (NGLDM).
