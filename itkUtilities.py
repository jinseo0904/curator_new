import SimpleITK as sitk
#import itk

# itkReorientImage()
# Set the voxel ordering for a 3D itkImage
#   img - 3D itkImage
#   code - 3 letter code for voxel order, e.g. "RAI"
# Return - 3D itkImage with desired voxel ordering
def reorientImage(img,code):

    # reorient image
    #ITK_COORDINATE_UNKNOWN = 0
    #ITK_COORDINATE_Right = 2
    #ITK_COORDINATE_Left = 3
    #ITK_COORDINATE_Posterior = 4
    #ITK_COORDINATE_Anterior = 5
    #ITK_COORDINATE_Inferior = 8
    #ITK_COORDINATE_Superior = 9

    #ITK_COORDINATE_PrimaryMinor = 0
    #ITK_COORDINATE_SecondaryMinor = 8
    #ITK_COORDINATE_TertiaryMinor = 16

    #ITK_COORDINATE_ORIENTATION_RAI = ( ITK_COORDINATE_Right << ITK_COORDINATE_PrimaryMinor ) \
    #    + ( ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor ) \
    #    + ( ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor )

    itkKey = { "R":2, "L":3, "P":4, "A":5, "I":8,"S":9,"PrimaryMinor":0,"SecondaryMinor":8,"TertiaryMinor":16}

    ITK_COORD =  (itkKey[code[0]] << itkKey["PrimaryMinor"])
    ITK_COORD += (itkKey[code[1]] << itkKey["SecondaryMinor"])
    ITK_COORD += (itkKey[code[2]] << itkKey["TertiaryMinor"])

    orienter = sitk.OrientImageFilter[type(img), type(img)].New()
    orienter.SetUseImageDirection(True)
    orienter.SetDesiredCoordinateOrientation(ITK_COORD)
    orienter.SetInput(img)
    orienter.Update()
    outImage = orienter.GetOutput()
    return(outImage)

# Note: to get equivalent result to windowLevelArray:
#  windowLevelArray(img, W, L) = windowLevelImage(img, W-1, L-0.5)
def windowLevelImage(img, window, level):

    filter = sitk.IntensityWindowingImageFilter[type(img),type(img)].New()
    filter.SetInput(img)
    #filter.SetWindowMinimum(-500)
    #filter.SetWindowMaximum(1299)
    filter.SetWindowLevel(window,level)
    filter.SetOutputMinimum(0)
    filter.SetOutputMaximum(255)
    filter.Update()
    outImg = filter.GetOutput()
    return(outImg)

def imageViewFromArray( array, template=None ):
    img = sitk.image_view_from_array( array )
    if not template is None:
        img.SetSpacing( template.GetSpacing() )
        img.SetOrigin( template.GetOrigin() )
        img.SetDirection( template.GetDirection() )

    return(img)

def resizeImage( img, outSize, outSpacing=None, outOrigin=None, interpolation="Linear" ):

    Dimension = img.GetDimension()
    inSize = img.GetSize()
    inSpacing = img.GetSpacing()
    inOrigin = img.GetOrigin()

    if outSpacing is None:
        outSpacing = [ y*inSize[x]/outSize[x] for x,y in enumerate(inSpacing) ]

    if outOrigin is None:
        outOrigin = inOrigin

    interpolator = None

    if interpolation=="Linear":
        interpolator = sitk.LinearInterpolateImageFunction.New(img)
    if interpolation=="Label":
        interpolator = sitk.LabelImageGaussianInterpolateImageFunction.New(img)

    if interpolator is None:
        print("ERROR: Invalid interpolation type: "+str(interpolation))
        return(None)

    transform = sitk.IdentityTransform[sitk.D,Dimension].New()

    resampled = sitk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=outSize,
        output_spacing=outSpacing,
        output_origin=outOrigin)

    return(resampled)



def resizeImageToReference( img, referenceImg ):

    filter = sitk.ResampleImageFilter[type(img), type(referenceImg)].New()

    Dimension = img.GetImageDimension()
    if Dimension != referenceImg.GetImageDimension():
        print("Images must be of the same dimension")
        return(None)

    inSize = sitk.size(img)
    inSpacing = img.GetSpacing()
    inOrigin = img.GetOrigin()

    outSize = sitk.size(referenceImg)
    outSpacing = referenceImg.GetSpacing()
    outOrigin = referenceImg.GetOrigin()
    outDirection = referenceImg.GetDirection()

    interpolator = sitk.LinearInterpolateImageFunction.New(img)
    transform = sitk.IdentityTransform[sitk.D,Dimension].New()

    resampled = sitk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=outSize,
        output_spacing=outSpacing,
        output_origin=outOrigin,
        output_direction=outDirection)

    return(resampled)

def compareImageHeaders(img1, img2, checkType=False):
    if sitk.size(img1) !=  sitk.size(img2):
        return False

    if img1.GetSpacing() != img2.GetSpacing():
        return False

    if img1.GetOrigin() != img2.GetOrigin():
        return False

    if img1.GetDirection() != img2.GetDirection():
        return False

    if checkType:
        if type(img1) != type(img2):
            return False

    return True

def getAxialSliceFromVolume( img, z ):
    filter = sitk.ExtractImageFilter.New(img)
    filter.SetDirectionCollapseToSubmatrix()

    region = img.GetBufferedRegion()
    size = region.GetSize()
    size[2] = 1
    start = region.GetIndex()
    start[2] = z
    outRegion = region
    outRegion.SetSize(size)
    outRegion.SetIndex(start)

    filter.SetExtractionRegion(outRegion)
    filter.Update()
    slice = filter.GetOutput()
    return(slice)
