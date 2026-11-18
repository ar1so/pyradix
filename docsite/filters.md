# Image filters

A filter turns the original image into one or more derived images; first-order and texture features are then
computed again on each derived image. Shape features are only computed on the original. Result keys name the image
they come from, for example `log-sigma-2-0-mm-3D_glcm_Contrast` or `wavelet-LLH_firstorder_Mean`.

Which filters exist depends on the mode, because IBSI and PyRadiomics define them differently.

| Filter | `ibsi` mode | `pyradiomics` mode | Key settings |
|---|---|---|---|
| **Original** | ✓ | ✓ | none |
| **LoG** (Laplacian of Gaussian) | IBSI 2 discrete kernel | SimpleITK recursive Gaussian, as PyRadiomics | `sigma` (mm, list) |
| **Wavelet** | IBSI 2 (Haar, db2, coif1) | stationary wavelet transform, as PyRadiomics | `wavelet` (default `coif1`), `start_level` (0), `level` (1) |
| **Mean** | IBSI 2 mean filter | – | `support` (odd, ≥ 1), `padding` |
| **Laws** | IBSI 2 Laws kernels with energy map | – | `kernel` (e.g. `E5L5S5`), `padding` |
| **Gabor** | IBSI 2 Gabor filter | – | `sigma`, `lambda`, `gamma`, `theta`, `padding` |
| **Square** | – | \(x^2\), rescaled as PyRadiomics | none |
| **SquareRoot** | – | \(\sqrt{\lvert x \rvert}\), sign kept | none |
| **Logarithm** | – | \(\log(\lvert x \rvert + 1)\), sign kept | none |
| **Exponential** | – | \(e^{x}\), rescaled as PyRadiomics | none |
| **Gradient** | – | gradient magnitude | `gradientUseSpacing` (default true) |
| **LBP2D** | – | local binary pattern per slice | `lbp2DRadius` (1), `lbp2DSamples` (8), `lbp2DMethod` (`uniform`) |
| **LBP3D** | – | 3D local binary pattern (spherical harmonics) | `lbp3DLevels` (2), `lbp3DIcosphereRadius` (1), `lbp3DIcosphereSubdivision` (1) |

## Enabling filters

In a PyRadiomics parameter file:

```yaml
imageType:
  Original: {}
  LoG:
    sigma: [1.0, 2.0, 3.0]
  Wavelet: {}
```

In Python:

```python
from pyradix import Params, RadixExtractor

params = Params(
    mode="pyradiomics",
    image_types=(("Original", {}), ("LoG", {"sigma": [1.0, 2.0, 3.0]}), ("Wavelet", {})),
)
result = RadixExtractor(mode="pyradiomics", params=params).extract("image.nii.gz", "mask.nii.gz")
```

## How the filters are checked

- **IBSI mode:** the Mean, LoG, Laws, Gabor and wavelet responses are compared with the IBSI 2 reference phantoms.
- **PyRadiomics mode:** Original, LoG (σ 1, 2, 3 mm) and all 8 wavelet bands give the same 1,209 feature values as
  PyRadiomics 3.1.0 on the PyRadiomics brain1 sample and the IBSI lung CT; Square, Square Root, Logarithm,
  Exponential, Gradient and LBP are compared with PyRadiomics in the test suite.

Details of each difference between the IBSI and PyRadiomics versions are on [Mode differences](deviations.md).
