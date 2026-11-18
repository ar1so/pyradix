# Shape (2D)

2D shape features describe the region in a single slice. They are computed when `force2D` is on (for example for
ultrasound frames or single-slice masks), using the slice orientation set by `force2Ddimension`.

The region's outline is turned into a closed polygon (marching squares). Let it have surface area \(A\) and
perimeter \(P\), and let \(\lambda_{major} \ge \lambda_{minor}\) be the PCA eigenvalues of the pixel coordinates.

| Feature | Definition |
|---|---|
| **MeshSurface** | \(A\), the area enclosed by the polygon |
| **PixelSurface** | \(N_p \, A_{pixel}\), the area counted in pixels |
| **Perimeter** | \(P\), the length of the polygon |
| **PerimeterSurfaceRatio** | \(P / A\) |
| **Sphericity** | \(\displaystyle \frac{2\sqrt{\pi A}}{P}\); 1 for a perfect circle |
| **SphericalDisproportion** *(deprecated)* | \(\displaystyle \frac{P}{2\sqrt{\pi A}}\), the inverse of Sphericity |
| **MaximumDiameter** | Largest Euclidean distance between two polygon vertices |
| **MajorAxisLength** | \(4\sqrt{\lambda_{major}}\) |
| **MinorAxisLength** | \(4\sqrt{\lambda_{minor}}\) |
| **Elongation** | \(\sqrt{\lambda_{minor} / \lambda_{major}}\) |
