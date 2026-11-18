# Shape (3D)

Shape features describe the size and form of the region. They depend only on the mask, not on the intensities, so
they are computed once, on the original (unfiltered) image.

Most are computed from a **triangle mesh** of the region's surface, made with marching cubes. Let the mesh have
volume \(V\) and surface area \(A\). Axis lengths use the principal component analysis (PCA) of the physical
coordinates of the ROI voxels, with eigenvalues \(\lambda_{major} \ge \lambda_{minor} \ge \lambda_{least}\).

| Feature | Definition |
|---|---|
| **MeshVolume** | \(\displaystyle V = \sum_{f} \frac{\mathbf{O}\mathbf{a}_f \cdot (\mathbf{O}\mathbf{b}_f \times \mathbf{O}\mathbf{c}_f)}{6}\), the signed volume of the tetrahedra formed by each mesh triangle \(f\) and the origin |
| **VoxelVolume** | \(N_p \, V_{voxel}\), the volume counted in voxels |
| **SurfaceArea** | \(\displaystyle A = \sum_{f} \tfrac{1}{2}\, \lvert \mathbf{a}_f\mathbf{b}_f \times \mathbf{a}_f\mathbf{c}_f \rvert\) |
| **SurfaceVolumeRatio** | \(A / V\) |
| **Sphericity** | \(\displaystyle \frac{\sqrt[3]{36 \pi V^2}}{A}\); 1 for a perfect sphere |
| **Compactness1** *(deprecated)* | \(\displaystyle \frac{V}{\sqrt{\pi A^3}}\) |
| **Compactness2** *(deprecated)* | \(\displaystyle \frac{36 \pi V^2}{A^3}\) |
| **SphericalDisproportion** *(deprecated)* | \(\displaystyle \frac{A}{\sqrt[3]{36 \pi V^2}}\), the inverse of Sphericity |
| **Maximum3DDiameter** | Largest Euclidean distance between two mesh vertices |
| **Maximum2DDiameterSlice** | Largest distance between two vertices in the same axial plane (row–column) |
| **Maximum2DDiameterColumn** | Largest distance between two vertices in the same coronal plane (row–slice) |
| **Maximum2DDiameterRow** | Largest distance between two vertices in the same sagittal plane (column–slice) |
| **MajorAxisLength** | \(4\sqrt{\lambda_{major}}\) |
| **MinorAxisLength** | \(4\sqrt{\lambda_{minor}}\) |
| **LeastAxisLength** | \(4\sqrt{\lambda_{least}}\) |
| **Elongation** | \(\sqrt{\lambda_{minor} / \lambda_{major}}\) |
| **Flatness** | \(\sqrt{\lambda_{least} / \lambda_{major}}\) |

!!! note "Differences between the modes"
    - **Mesh.** `pyradiomics` mode builds the mesh exactly as PyRadiomics' `cshape.c` does (a 128-case marching-cubes
      table); `ibsi` mode uses the standard 256-case Lorensen table at the 0.5 iso-level. The two meshes differ slightly,
      so MeshVolume, SurfaceArea and the features derived from them can differ in the fourth significant digit.
    - **PCA.** `ibsi` mode divides the covariance by \(N-1\), as the IBSI digital phantom requires; `pyradiomics` mode
      divides by \(N\), as PyRadiomics does.
    - The 2D diameters are not defined by IBSI; both modes use PyRadiomics' rule (vertices count as "in the same plane"
      when their coordinate on the dropped axis is equal).
