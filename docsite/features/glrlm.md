# GLRLM: grey-level run-length matrix

A **run** is a line of consecutive voxels with the same grey level along one direction. \(\mathbf{P}(i,j)\) is the
number of runs of grey level \(i\) and length \(j\). Features are computed per direction (13 in 3D) and averaged.

Notation: \(N_r = \sum_{i,j}\mathbf{P}(i,j)\) is the number of runs, \(N_p\) the number of voxels,
\(p(i,j) = \mathbf{P}(i,j)/N_r\), and \(\mu_i = \sum_{i,j} p(i,j)\,i\), \(\mu_j = \sum_{i,j} p(i,j)\,j\).

| Feature | Definition |
|---|---|
| **ShortRunEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{j^2}\) |
| **LongRunEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \mathbf{P}(i,j)\, j^2\) |
| **GrayLevelNonUniformity** | \(\dfrac{1}{N_r}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **GrayLevelNonUniformityNormalized** | \(\dfrac{1}{N_r^2}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **RunLengthNonUniformity** | \(\dfrac{1}{N_r}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **RunLengthNonUniformityNormalized** | \(\dfrac{1}{N_r^2}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **RunPercentage** | \(N_r / N_p\) |
| **GrayLevelVariance** | \(\sum_{i,j} p(i,j)\,(i - \mu_i)^2\) |
| **RunVariance** | \(\sum_{i,j} p(i,j)\,(j - \mu_j)^2\) |
| **RunEntropy** | \(-\sum_{i,j} p(i,j) \log_2\big(p(i,j)+\epsilon\big)\) |
| **LowGrayLevelRunEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2}\) |
| **HighGrayLevelRunEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \mathbf{P}(i,j)\, i^2\) |
| **ShortRunLowGrayLevelEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2 j^2}\) |
| **ShortRunHighGrayLevelEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, i^2}{j^2}\) |
| **LongRunLowGrayLevelEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, j^2}{i^2}\) |
| **LongRunHighGrayLevelEmphasis** | \(\dfrac{1}{N_r}\sum_{i,j} \mathbf{P}(i,j)\, i^2 j^2\) |
