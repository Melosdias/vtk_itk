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


# Segmentation

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
seedY, seedX = plt.ginput()
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
