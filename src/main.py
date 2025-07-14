import itk
import matplotlib.pyplot as plt

fixed_image = itk.imread("Data/case6_gre1.nrrd", itk.F)
moving_image = itk.imread("Data/case6_gre2.nrrd", itk.F)

dimension = 3
FixedImageType = type(fixed_image)
MovingImageType = type(moving_image)


TransformType = itk.TranslationTransform[itk.D, dimension]
transform = TransformType.New()

optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
    LearningRate=1.0,
    MinimumStepLength=0.001,
    NumberOfIterations=200,
)


metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New(
    Metric=metric,
    Optimizer=optimizer,
    FixedImage=fixed_image,
    MovingImage=moving_image,
    InitialTransform=transform,
)

registration.Update()

resampler = itk.ResampleImageFilter.New(
    Input=moving_image,
    Transform=registration.GetTransform(),
    UseReferenceImage=True,
    ReferenceImage=fixed_image,
    DefaultPixelValue=0,
)
resampled_image = resampler.Update()

itk.imwrite(resampler, "aligned.nrrd")


# Segmentation image 1

#if ginput dos not work 
seedX=110
seedY=100
lower=190 
upper=255

input_image = itk.imread("aligned.nrrd", pixel_type=itk.F)

smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image, NumberOfIterations=20, TimeStep=0.04,
                                                           ConductanceParameter=3)

smoother.Update()
smoothed_image = smoother.GetOutput()

plt.ion()
plt.imshow(smoother.GetOutput()[0], cmap="gray")
seedY, seedX = plt.ginput()[0]
seedX, seedY = int(seedX), int(seedY)
print("Seed coordinates : ", seedX, seedY)


z = 0  
initial_value = smoothed_image.GetPixel((seedX, seedY, z))
lower = initial_value - 10
upper = initial_value + 30


connected_threshold = itk.ConnectedThresholdImageFilter.New(smoothed_image)
connected_threshold.SetReplaceValue(255)
connected_threshold.SetLower(lower)
connected_threshold.SetUpper(upper)

connected_threshold.SetSeed((seedX, seedY, z))
connected_threshold.Update()
plt.ion()
plt.imshow(itk.GetArrayViewFromImage(connected_threshold.GetOutput())[0], cmap="gray")

dimension = input_image.GetImageDimension()

in_type = itk.output(connected_threshold)
output_type = itk.Image[itk.UC, dimension]
rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)
rescaler.Update()

output_filepath = "segmentation.nrrd"
itk.imwrite(rescaler, output_filepath)


# Segmentation image 2

gre1 = itk.imread("./Data/case6_gre2.nrrd", itk.F)
smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=gre1, NumberOfIterations=20, TimeStep=0.04,
                                                           ConductanceParameter=3)

smoother.Update()
smoothed_image = smoother.GetOutput()

fixed_array = itk.GetArrayViewFromImage(gre1)
z = fixed_array.shape[0] //2

print("Seed coordinates : ", seedX, seedY, z)

# Instantiate the filter


initial_value = smoothed_image.GetPixel((seedX, seedY, z))
lower = initial_value - 10
upper = initial_value + 30

print("initial value : ", smoothed_image.GetPixel((seedX, seedY, z)))
print("lower, upper : ", lower, upper)

# Configure filter from the command line arguments
connected_threshold = itk.ConnectedThresholdImageFilter.New(smoothed_image)
connected_threshold.SetReplaceValue(255)
connected_threshold.SetLower(lower)
connected_threshold.SetUpper(upper)

connected_threshold.SetSeed((seedX, seedY, z))
connected_threshold.Update()
plt.ion()
seg_array = itk.GetArrayViewFromImage(connected_threshold.GetOutput())
plt.imshow(seg_array[z, :, :], cmap="gray")

dimension = gre1.GetImageDimension()

in_type = itk.output(connected_threshold)
output_type = itk.Image[itk.UC, dimension]
rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold)
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)
rescaler.Update()

output_filepath = "segmentation2.nrrd"
itk.imwrite(rescaler, output_filepath)


# Affichage

import vtk

# Load the volume
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.1, 0.1)

renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
renwin.SetInteractor(interactor)

reader = vtk.vtkNrrdReader()
reader.SetFileName("./aligned.nrrd")
reader.Update()

# Affichage du scan en transparent
opacityFun = vtk.vtkPiecewiseFunction()
opacityFun.AddPoint(0, 0.0)
opacityFun.AddPoint(100, 0.0)
opacityFun.AddPoint(200, 0.05)
opacityFun.AddPoint(210, 0.0)
opacityFun.AddPoint(255, 0.0)

colorFun = vtk.vtkColorTransferFunction()
colorFun.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorFun.AddRGBPoint(200, 1.0, 1.0, 1.0)
colorFun.AddRGBPoint(255, 1.0, 1.0, 1.0)

property = vtk.vtkVolumeProperty()
property.SetColor(colorFun)
property.SetScalarOpacity(opacityFun)
property.SetInterpolationTypeToLinear()

mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputConnection(reader.GetOutputPort())

volume = vtk.vtkVolume()
volume.SetProperty(property)
volume.SetMapper(mapper)


# Affichage de la tumeur sur la deuxième segmentation
reader3 = vtk.vtkNrrdReader()
reader3.SetFileName("./segmentation2.nrrd")
reader3.Update()

opacityTumor2 = vtk.vtkPiecewiseFunction()
opacityTumor2.AddPoint(0, 0.0)
opacityTumor2.AddPoint(100, 0.1)
opacityTumor2.AddPoint(255, 0.0)

colorTumor2 = vtk.vtkColorTransferFunction()
colorTumor2.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorTumor2.AddRGBPoint(255, 0.0, 1.0, 0.0)

propertyTumor2 = vtk.vtkVolumeProperty()
propertyTumor2.SetColor(colorTumor2)
propertyTumor2.SetScalarOpacity(opacityTumor2)
propertyTumor2.SetInterpolationTypeToLinear()

mapperTumor2 = vtk.vtkSmartVolumeMapper()
mapperTumor2.SetInputConnection(reader3.GetOutputPort())

volumeTumor2 = vtk.vtkVolume()
volumeTumor2.SetProperty(propertyTumor2)
volumeTumor2.SetMapper(mapperTumor2)

# Affichage de la tumeur sur la première segmentation
reader2 = vtk.vtkNrrdReader()
reader2.SetFileName("./segmentation.nrrd")
reader2.Update()

opacityTumor = vtk.vtkPiecewiseFunction()
opacityTumor.AddPoint(0, 0.0)
opacityTumor.AddPoint(255, 1.0)

colorTumor = vtk.vtkColorTransferFunction()
colorTumor.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorTumor.AddRGBPoint(255, 0.0, 0.0, 1.0)

propertyTumor = vtk.vtkVolumeProperty()
propertyTumor.SetColor(colorTumor)
propertyTumor.SetScalarOpacity(opacityTumor)
propertyTumor.SetInterpolationTypeToLinear()

mapperTumor = vtk.vtkSmartVolumeMapper()
mapperTumor.SetInputConnection(reader2.GetOutputPort())

volumeTumor = vtk.vtkVolume()
volumeTumor.SetProperty(propertyTumor)
volumeTumor.SetMapper(mapperTumor)


renderer.AddVolume(volume)
renderer.AddVolume(volumeTumor2)
renderer.AddVolume(volumeTumor)

renwin.Render()
interactor.Start()
