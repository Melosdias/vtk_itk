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

# Reading images

# First set of images
print("Reading Data/case6_gre1.nrrd...")

input_image1 = itk.imread("Data/case6_gre1.nrrd", pixel_type=itk.F)
smoother1 = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image1, NumberOfIterations=20, TimeStep=0.04,
                                                            ConductanceParameter=3)

smoother1.Update()
smoothed_image1 = smoother1.GetOutput()

y_offset = 15
# Crop the bottom part
crop1 = itk.CropImageFilter.New(Input=smoothed_image1)
crop1.SetLowerBoundaryCropSize([0, 0, 0])
crop1.SetUpperBoundaryCropSize([0, y_offset, 0])
crop1.Update()

itk.imwrite(crop1, "cropped_case6_gre1.nrrd")

cropped_smoothed_image1 = crop1.GetOutput()
cropped_smoothed_image_array1 = itk.GetArrayViewFromImage(cropped_smoothed_image1)

# Second set of images
print("Reading Data/case6_gre2.nrrd...")

input_image2 = itk.imread("./Data/case6_gre2.nrrd", itk.F)
smoother2 = itk.GradientAnisotropicDiffusionImageFilter.New(Input=input_image2, NumberOfIterations=20, TimeStep=0.04,
                                                            ConductanceParameter=3)
smoother2.Update()
smoothed_image2 = smoother2.GetOutput()

# y_offset = 20
# Crop the top part
crop2 = itk.CropImageFilter.New(Input=smoothed_image2)
crop2.SetLowerBoundaryCropSize([0, y_offset, 0])
crop2.SetUpperBoundaryCropSize([0, 0, 0])
crop2.Update()

cropped_smoothed_image2 = crop2.GetOutput()
cropped_smoothed_image_array2 = itk.GetArrayViewFromImage(cropped_smoothed_image2)

print(cropped_smoothed_image_array1.shape, cropped_smoothed_image_array2.shape)
assert (cropped_smoothed_image_array1.shape == cropped_smoothed_image_array2.shape)

# User interaction part

seedX = 110
seedY = 100
lower = 190
upper = 255

current_z = cropped_smoothed_image_array1.shape[0] // 2
seed_coords = None
seed_marker1 = None
seed_marker2 = None

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

im1 = ax[0].imshow(cropped_smoothed_image_array1[current_z], cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("First set of images")
im2 = ax[1].imshow(cropped_smoothed_image_array2[current_z], cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Second set of images")

fig.suptitle(f"Select seed point - Slice {current_z}")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = plt.Slider(ax_slider, 'Z-slice', 0, cropped_smoothed_image_array1.shape[0] - 1,
                    valinit=current_z, valfmt='%d')


def update_slice(val):
    global current_z, seed_marker1, seed_marker2
    current_z = int(slider.val)
    im1.set_array(cropped_smoothed_image_array1[current_z])
    im2.set_array(cropped_smoothed_image_array2[current_z])
    fig.suptitle(f"Seed selected at ({seedX}, {seedY}) - Slice {current_z}")

    # Clear seed marker when changing slices
    if seed_marker1:
        seed_marker1.remove()
        seed_marker1 = None
    if seed_marker2:
        seed_marker2.remove()
        seed_marker2 = None

    fig.canvas.draw()


def on_click(event):
    global seed_coords, seed_marker1, seed_marker2
    if event.button != 1 or event.xdata is None or event.ydata is None:
        return

    # print(event)

    x, y = int(event.xdata), int(event.ydata)
    seed_coords = (x, y, current_z)

    # Clear seed marker when changing slices
    if seed_marker1:
        seed_marker1.remove()
        seed_marker1 = None
    if seed_marker2:
        seed_marker2.remove()
        seed_marker2 = None

    # Add new marker
    seed_marker1 = ax[0].plot(x, y, 'r+')[0]
    seed_marker2 = ax[1].plot(x, y, 'r+')[0]

    fig.suptitle(f"Seed selected at ({x}, {y}) - Slice {current_z}")
    fig.canvas.draw()

    # print(x, y, current_z)


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

# Old way
# seedX, seedY = plt.ginput()[0]

print("Waiting for the segmentation of the images...")

# Segmentation image 1

initial_value1 = cropped_smoothed_image1.GetPixel((seedX, seedY, seedZ))
lower1 = initial_value1 - 10
upper1 = initial_value1 + 30

print("initial_value1 : ", initial_value1)
print("lower1, upper1 : ", lower1, upper1)

connected_threshold1 = itk.ConnectedThresholdImageFilter.New(Input=cropped_smoothed_image1)
connected_threshold1.SetReplaceValue(255)
connected_threshold1.SetLower(lower1)
connected_threshold1.SetUpper(upper1)

connected_threshold1.SetSeed((seedX, seedY, seedZ))
connected_threshold1.Update()

dimension1 = input_image1.GetImageDimension()

in_type1 = itk.output(connected_threshold1)
output_type1 = itk.Image[itk.UC, dimension1]
segmentation_image_rescaler1 = itk.RescaleIntensityImageFilter[in_type1, output_type1].New(Input=connected_threshold1)
segmentation_image_rescaler1.SetOutputMinimum(0)
segmentation_image_rescaler1.SetOutputMaximum(255)
segmentation_image_rescaler1.Update()

output_filepath1 = "segmentation.nrrd"
itk.imwrite(segmentation_image_rescaler1, output_filepath1)

# Segmentation image 2

initial_value2 = cropped_smoothed_image2.GetPixel((seedX, seedY, seedZ))
lower2 = initial_value2 - 10
upper2 = initial_value2 + 30

print("initial_value2 : ", initial_value2)
print("lower2, upper2 : ", lower2, upper2)

# Configure filter from the command line arguments
connected_threshold2 = itk.ConnectedThresholdImageFilter.New(Input=cropped_smoothed_image2)
connected_threshold2.SetReplaceValue(255)
connected_threshold2.SetLower(lower2)
connected_threshold2.SetUpper(upper2)

connected_threshold2.SetSeed((seedX, seedY, seedZ))
connected_threshold2.Update()

dimension2 = input_image2.GetImageDimension()

in_type2 = itk.output(connected_threshold2)
output_type2 = itk.Image[itk.UC, dimension2]
segmentation_image_rescaler2 = itk.RescaleIntensityImageFilter[in_type2, output_type2].New(Input=connected_threshold2)
segmentation_image_rescaler2.SetOutputMinimum(0)
segmentation_image_rescaler2.SetOutputMaximum(255)
segmentation_image_rescaler2.Update()

output_filepath2 = "segmentation2.nrrd"
itk.imwrite(segmentation_image_rescaler2, output_filepath2)

# Show the segmented images

segmented_image1 = itk.GetArrayViewFromImage(segmentation_image_rescaler1.GetOutput())
segmented_image2 = itk.GetArrayViewFromImage(segmentation_image_rescaler2.GetOutput())

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

im1 = ax[0].imshow(segmented_image1[current_z], cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("First set of images")
seed_marker1 = ax[0].plot(seedX, seedY, 'r+')[0]
im2 = ax[1].imshow(segmented_image2[current_z], cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Second set of images")
seed_marker2 = ax[1].plot(seedX, seedY, 'r+')[0]

fig.suptitle(f"Seed selected at ({seedX}, {seedY}), first and second segmentation - Slice {current_z}")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = plt.Slider(ax_slider, 'Z-slice', 0, segmented_image1.shape[0] - 1,
                    valinit=current_z, valfmt='%d')


def update_slice2(val):
    current_z = int(slider.val)
    im1.set_array(segmented_image1[current_z])
    im2.set_array(segmented_image2[current_z])
    fig.suptitle(f"Seed selected at ({seedX}, {seedY}), first and second segmentation - Slice {current_z}")
    fig.canvas.draw()


# Connect events
slider.on_changed(update_slice2)

# Instructions
fig.text(0.5, 0.02, 'Use slider to navigate slices, close window when done',
         ha='center', fontsize=10)

plt.show()

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
reader.SetFileName("./cropped_case6_gre1.nrrd")
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

# Add all the volumes

renderer.AddVolume(volume)
renderer.AddVolume(volumeTumor)
renderer.AddVolume(volumeTumor2)

renwin.Render()
interactor.Start()
