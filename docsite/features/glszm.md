# GLSZM: grey-level size-zone matrix

A **zone** is a group of connected voxels with the same grey level (26-connected in 3D, 8-connected in 2D).
\(\mathbf{P}(i,j)\) is the number of zones of grey level \(i\) and size \(j\) voxels. Unlike GLCM and GLRLM, the
GLSZM does not depend on direction.

Notation: \(N_z = \sum_{i,j}\mathbf{P}(i,j)\) is the number of zones, \(N_p\) the number of voxels,
\(p(i,j) = \mathbf{P}(i,j)/N_z\), and \(\mu_i = \sum_{i,j} p(i,j)\,i\), \(\mu_j = \sum_{i,j} p(i,j)\,j\).

| Feature | Definition |
|---|---|
| **SmallAreaEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{j^2}\) |
| **LargeAreaEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, j^2\) |
| **GrayLevelNonUniformity** | \(\dfrac{1}{N_z}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **GrayLevelNonUniformityNormalized** | \(\dfrac{1}{N_z^2}\sum_{i} \Big(\sum_{j} \mathbf{P}(i,j)\Big)^2\) |
| **SizeZoneNonUniformity** | \(\dfrac{1}{N_z}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **SizeZoneNonUniformityNormalized** | \(\dfrac{1}{N_z^2}\sum_{j} \Big(\sum_{i} \mathbf{P}(i,j)\Big)^2\) |
| **ZonePercentage** | \(N_z / N_p\) |
| **GrayLevelVariance** | \(\sum_{i,j} p(i,j)\,(i - \mu_i)^2\) |
| **ZoneVariance** | \(\sum_{i,j} p(i,j)\,(j - \mu_j)^2\) |
| **ZoneEntropy** | \(-\sum_{i,j} p(i,j) \log_2\big(p(i,j)+\epsilon\big)\) |
| **LowGrayLevelZoneEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2}\) |
| **HighGrayLevelZoneEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, i^2\) |
| **SmallAreaLowGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)}{i^2 j^2}\) |
| **SmallAreaHighGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, i^2}{j^2}\) |
| **LargeAreaLowGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \dfrac{\mathbf{P}(i,j)\, j^2}{i^2}\) |
| **LargeAreaHighGrayLevelEmphasis** | \(\dfrac{1}{N_z}\sum_{i,j} \mathbf{P}(i,j)\, i^2 j^2\) |
