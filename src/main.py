# ITK part

import itk
import matplotlib.pyplot as plt


# fixed_image = itk.imread("Data/case6_gre1.nrrd", itk.F)
# moving_image = itk.imread("Data/case6_gre2.nrrd", itk.F)
#
# dimension = 3
# FixedImageType = type(fixed_image)
# MovingImageType = type(moving_image)
#
#
# TransformType = itk.TranslationTransform[itk.D, dimension]
# transform = TransformType.New()
#
# optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
#     LearningRate=1.0,
#     MinimumStepLength=0.001,
#     NumberOfIterations=200,
# )
#
#
# metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()
#
# registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New(
#     Metric=metric,
#     Optimizer=optimizer,
#     FixedImage=fixed_image,
#     MovingImage=moving_image,
#     InitialTransform=transform,
# )
#
# registration.Update()
#
# resampler = itk.ResampleImageFilter.New(
#     Input=moving_image,
#     Transform=registration.GetTransform(),
#     UseReferenceImage=True,
#     ReferenceImage=fixed_image,
#     DefaultPixelValue=0,
# )
# resampled_image = resampler.Update()
#
# itk.imwrite(resampler, "aligned.nrrd")

# Segmentation image 1

# If ginput does not work
seedX = 110
seedY = 100
lower = 190
upper = 255

print("Reading Data/case6_gre1.nrrd...")

# input_image = itk.imread("aligned.nrrd", pixel_type=itk.F)
input_image1 = itk.imread("Data/case6_gre1.nrrd", pixel_type=itk.F)
print(itk.GetArrayViewFromImage(input_image1).shape)

smoother1 = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image1, NumberOfIterations=20, TimeStep=0.04,
                                                            ConductanceParameter=3)

smoother1.Update()
smoothed_image1 = smoother1.GetOutput()

smoothed_image_array = itk.GetArrayViewFromImage(smoothed_image1)
current_z = smoothed_image_array.shape[0] // 2
seed_coords = None
seed_marker = None

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

im = ax.imshow(smoothed_image_array[current_z], cmap="gray")
ax.set_title(f"Select seed point - Slice {current_z}")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = plt.Slider(ax_slider, 'Z-slice', 0, smoothed_image_array.shape[0]-1, 
               valinit=current_z, valfmt='%d')

def update_slice(val):
    global current_z, seed_marker
    current_z = int(slider.val)
    im.set_array(smoothed_image_array[current_z])
    ax.set_title(f"Select seed point - Slice {current_z}")
    
    # Clear seed marker when changing slices
    if seed_marker:
        seed_marker.remove()
        seed_marker = None
    
    fig.canvas.draw()

def on_click(event):
    global seed_coords, seed_marker
    if event.inaxes != ax or event.button != 1:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    seed_coords = (x, y, current_z)
    
    # Clear previous marker
    if seed_marker:
        seed_marker.remove()
    
    # Add new marker
    seed_marker = ax.plot(x, y, 'ro', markersize=8, markerfacecolor='none', 
                         markeredgewidth=2)[0]
    
    ax.set_title(f"Seed selected at ({x}, {y}) - Slice {current_z}")
    fig.canvas.draw()
    print(f"Seed selected: x={x}, y={y}, z={current_z}")

# Connect events
slider.on_changed(update_slice)
fig.canvas.mpl_connect('button_press_event', on_click)

# Instructions
fig.text(0.5, 0.02, 'Use slider to navigate slices, left-click to select seed, close window when done', 
         ha='center', fontsize=10)

plt.show()

# Get final coordinates
if seed_coords:
    seedX, seedY, seedZ = seed_coords
    print(f"Final seed coordinates: X={seedX}, Y={seedY}, Z={seedZ}")
else:
    print("No seed selected, using defaults")
    seedZ = current_z
    print(f"Using Z slice: {seedZ}")


# Show first image to select the seed
plt.ion()
title = "Waiting for the user to chose a seed (left click)..."
plt.title(title)
plt.imshow(smoother1.GetOutput()[seedZ], cmap="gray")

print(title)

seedX, seedY = plt.ginput()[0]
seedX, seedY = int(seedX), int(seedY)
print("Seed coordinates : ", seedX, seedY, z)

plt.ioff()
plt.title("Seed selected, waiting for the user to close the window")
plt.plot([seedX], [seedY], "r+")
plt.show()

print("Waiting for the segmentation of the images...")

initial_value = smoothed_image1.GetPixel((seedX, seedY, z))
lower = initial_value - 10
upper = initial_value + 30

print("initial value1 : ", initial_value)
print("lower1, upper1 : ", lower, upper)

connected_threshold1 = itk.ConnectedThresholdImageFilter.New(smoothed_image1)
connected_threshold1.SetReplaceValue(255)
connected_threshold1.SetLower(lower)
connected_threshold1.SetUpper(upper)

connected_threshold1.SetSeed((seedX, seedY, z))
connected_threshold1.Update()

dimension1 = input_image1.GetImageDimension()

in_type = itk.output(connected_threshold1)
output_type = itk.Image[itk.UC, dimension1]
segmentation_image_rescaler1 = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold1)
segmentation_image_rescaler1.SetOutputMinimum(0)
segmentation_image_rescaler1.SetOutputMaximum(255)
segmentation_image_rescaler1.Update()

output_filepath1 = "segmentation.nrrd"
itk.imwrite(segmentation_image_rescaler1, output_filepath1)

# Segmentation image 2

print("Reading Data/case6_gre2.nrrd...")

input_image2 = itk.imread("./Data/case6_gre2.nrrd", itk.F)
print(itk.GetArrayViewFromImage(input_image2).shape)

smoother2 = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image2, NumberOfIterations=20, TimeStep=0.04,
                                                            ConductanceParameter=3)
smoother2.Update()
smoothed_image2 = smoother2.GetOutput()

fixed_array = itk.GetArrayViewFromImage(input_image2)
z = fixed_array.shape[0] // 2

print("Seed coordinates : ", seedX, seedY, z)

# Instantiate the filter


initial_value = smoothed_image2.GetPixel((seedX, seedY, z))
lower = initial_value - 10
upper = initial_value + 30

print("initial value2 : ", initial_value)
print("lower2, upper2 : ", lower, upper)

# Configure filter from the command line arguments
connected_threshold2 = itk.ConnectedThresholdImageFilter.New(smoothed_image2)
connected_threshold2.SetReplaceValue(255)
connected_threshold2.SetLower(lower)
connected_threshold2.SetUpper(upper)

connected_threshold2.SetSeed((seedX, seedY, z))
connected_threshold2.Update()

# Show the segmented image
plt.title("Seed selected, first and second segmentation")
plt.subplot(2, 2, 1)
plt.imshow(itk.GetArrayViewFromImage(connected_threshold1.GetOutput())[z], cmap="gray")
plt.plot([seedX], [seedY], "r+")
plt.subplot(2, 2, 2)
plt.imshow(itk.GetArrayViewFromImage(connected_threshold2.GetOutput())[z], cmap="gray")
plt.plot([seedX], [seedY], "r+")

plt.show()

dimension2 = input_image2.GetImageDimension()

in_type = itk.output(connected_threshold2)
output_type = itk.Image[itk.UC, dimension2]
segmentation_image_rescaler2 = itk.RescaleIntensityImageFilter[in_type, output_type].New(connected_threshold2)
segmentation_image_rescaler2.SetOutputMinimum(0)
segmentation_image_rescaler2.SetOutputMaximum(255)
segmentation_image_rescaler2.Update()

output_filepath2 = "segmentation2.nrrd"
itk.imwrite(segmentation_image_rescaler2, output_filepath2)

# VTK part

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
reader.SetFileName("./Data/case6_gre1.nrrd")
reader.Update()

# Affichage du scan en transparent
opacityFun = vtk.vtkPiecewiseFunction()
opacityFun.AddPoint(0, 0.0)
opacityFun.AddPoint(40, 0.0)
opacityFun.AddPoint(100, 0.01)
opacityFun.AddPoint(150, 0.03)
opacityFun.AddPoint(200, 0.08)
opacityFun.AddPoint(255, 0.1)

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
opacityTumor.AddPoint(100, 0.4)
opacityTumor.AddPoint(255, 0.5)

colorTumor = vtk.vtkColorTransferFunction()
colorTumor.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorTumor.AddRGBPoint(255, 0.0, 0.0, 1.0)

propertyTumor = vtk.vtkVolumeProperty()
propertyTumor.SetColor(colorTumor)
propertyTumor.SetScalarOpacity(opacityTumor)
propertyTumor.SetInterpolationTypeToLinear()

# propertyTumor.ShadeOn()
# propertyTumor.SetAmbient(0.3)
# propertyTumor.SetDiffuse(0.6)
# propertyTumor.SetSpecular(0.1)

mapperTumor = vtk.vtkSmartVolumeMapper()
mapperTumor.SetInputConnection(reader2.GetOutputPort())

volumeTumor = vtk.vtkVolume()
volumeTumor.SetProperty(propertyTumor)
volumeTumor.SetMapper(mapperTumor)

renderer.AddVolume(volume)
renderer.AddVolume(volumeTumor)
renderer.AddVolume(volumeTumor2)

renwin.Render()
interactor.Start()
