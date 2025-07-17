# ITK/VTK Project

## Build

The project has been created using `itk-5.4.4` and `vtk-9.5.0` Python libraries.

To install them and run the script, you may use the provided `pyproject.toml` as well as the `uv.lock` and execute the
command:

```bash
uv run src/main.py
```

## Technical choices

### Aligning the images

The `Data` contains two files:

- `case6_gre1.nrrd`
- `case6_gre2.nrrd`

The first goal is to align both 3D images. To do this, we make use of the `itk.ResampleImageFilter` object with the
following parameters:

- transform: the transform of the registration `itk.ImageRegistrationMethodv4` with parameters:
    - metric: Mean Squared Error, provided by `itk.MeanSquaresImageToImageMetricv4`,
    - optimizer: uses a regular step gradient descent provided by `itk.RegularStepGradientDescentOptimizer`, with
      parameters `LearningRate = 1.0, MinimumStepLength = 1e-3, NumberOfIterations = 200`.
    - initial transform: a translation
- use reference image,
- fill with black pixels by default.

### Smoothing

The next step is to smooth the previously obtained images. The `itk.GradientAnisotropicDiffusionImageFilter` class has
been used for both images, with parameters:

- `NumberOfIterations = 20`
- `TimeStep = 0.04`
- `ConductanceParameter = 3`

### Selecting the seed

A first window, displayed with `matplotlib`, requests the user to select one of the 175 slices and a seed by
left-clicking on a specific pixel.
Both 3D images are side-to-side and selecting a pixel from one 3D image selects the same pixel for the other.

### Images segmentation

For both images, a histogram normalization is computed using `itk.ConnectedThresholdImageFilter` and
`itk.RescaleIntensityImageFilter` with parameters:

- connected threshold:
    - lower: seed value - 10
    - upper: seed value + 30
- rescale intensity filter:
    - output minimum: 0
    - output maximum: 255

This allows for a zone next to the seed point and close (in terms of pixel intensity) to the seed point to be replaced
by white, and the rest is in black.

The result is then displayed with `matplotlib.pyplot`.

### Visualization with `vtk`

We first set up a 3D scene environment with a basic renderer and render window.
Then we enable an interactive visualisation with
```interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())```

With this we are able to interact with the scene we want to display (rotate, translate, zoom...).

We will display 3 volumes, they are respectively:

- The outline of the head (white and transparent)
- The segmentation of the selected seed (tumor) of the first 3D image (blue)
- The segmentation of the selected seed (tumor) of the second 3D image (green)

We use the same concept to create the 3 volumes.

We use ```property = vtk.vtkVolumeProperty()``` to add the properties we want to the volume, Here are a few of them we
used:

- With ```opacityFun = vtk.vtkPiecewiseFunction()``` we can add points and value to the opacity function
  in order to have an opaque or transparent render of the volume.
- With ```colorFun = vtk.vtkColorTransferFunction()``` we can modify the color of the volume (in our case, white, blue
  and green)
- With ```property.SetInterpolationTypeToLinear()``` we can add a linear interpolation for the volume, it is also
  possible to use "nearest" instead
- With ```property.ShadeOn()```, ```property.SetAmbient(...)``` and others we can add usual properties for 3D rendering:
  like ambient, specular and diffuse values

Then we use the properties of the volume we want to render with the correct data.
For that we use ```vtk.vtkNrrdReader()``` to read the __.nrrd__ files (3D images), we created temporary ones to simplify
the ITK to VTK conversions.

After the creation of the 3 volumes we link them together thanks to the renderer ```renderer.AddVolume(...)``` and
finally call the event loop for the user interaction.

## Technical problems encountered

We encountered a few problems:

- Aligning the two 3D images was pretty weird, we tried cropping and extending the images for the itk part, and setting
  up new origins and directions for the metadata of the files for the vtk part... But we finally found a more robust and
  clean method (see __Aligning the images__ section)


- Finding a correct way to add an interactive and usable input method for the seed selection was not easy, we finally
  decided to use _matplotlib_ for that because we didn't succeed with vtk easily unfortunately


- Converting the images from the itk format to the vtk format was also very tedious. We didn't find a clean way to
  handle it, so we decided to use the simplicity of the __vtkNrrdReader__ by writing temporary files


- The segmentation also took some time, but found a very simple way to do it for both 3D images

# Authors

Emmeline Heitzler <emmeline.heitzler@epita.fr>

Julie Fiadino <julie.fiadino@epita.fr>

Maxime Legros <maxime.legros@epita.fr>

Melwan Chevassus <melwan.chevassus@epita.fr>


