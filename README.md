# ITK/VTK Project

## Build

The project has been created using `itk-5.4.4` and `vtk-9.5.0` Python libraries.

To install them and run the script, you may use the provided `pyproject.toml` as well as the `uv.lock` and execute the command:

```python
uv run src/main.py
```

## Technical choices

### Aligning the images

The `Data` contains two files:

- `case6_gre1.nrrd`
- `case6_gre2.nrrd`

The first goal is to align both 3D images. To do this, we make use of the `itk.ResampleImageFilter` object with the following parameters:

- transform: the transform of the registration `itk.ImageRegistrationMethodv4` with parameters:
    - metric: Mean Squared Error, provided by `itk.MeanSquaresImageToImageMetricv4`,
    - optimizer: uses a regular step gradient descent provided by `itk.RegularStepGradientDescentOptimizer`, with parameters `LearningRate = 1.0, MinimumStepLength = 1e-3, NumberOfIterations = 200`.
    - initial transform: a translation
- use reference image,
- fill with black pixels by default.

### Smoothing

The next step is to smooth the previously obtained images. The `itk.GradientAnisotropicDiffusionImageFilter` class has been used for both images, with parameters:

- `NumberOfIterations = 20`
- `TimeStep = 0.04`
- `ConductanceParameter = 3`

### Selecting the seed

A first window, displayed with `matplotlib`, requests the user to select one of the 175 slices. Then, on another window, it prompts the user to select a point in this slice so that it can be set as the seed.

### Images segmentation

For both images, a histogram normalization is computed using `itk.ConnectedThresholdImageFilter` and `itk.RescaleIntensityImageFilter` with parameters:

- connected threshold:
    - lower: seed value - 10
    - upper: seed value + 30
- rescale intensity filter:
    - output minimum: 0
    - output maximum: 255

This allows for a zone next to the seed point and close (in terms of pixel intensity) to the seed point to be replaced by white, and the rest is in black.

The result is then display with `matplotlib.pyplot`.