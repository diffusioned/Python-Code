# -*- coding: utf-8 -*-
# Python file to design best path(s) through an optical chip based on input and output positions
#
# Copyright Ned Charles 2011
# The references here are all used for education purposes only

############################################################################################################
# ** CLASS DECLARATION **
############################################################################################################
#
# PositionDimensionDataObject
#
# Contains all start and end positions, physical constraints and dimensions, 

class PositionDimensionDataObject(object):

    def __init__(self, NumberOfSegments):
        self.NumberOfSegments = NumberOfSegments
        self.PathXStartPosArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.PathYStartPosArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.PathXEndPosArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.PathYEndPosArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.XDistanceArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.YDistanceArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.DirectDistanceArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.SecondaryIndexList = np.zeros(1, dtype=np.int)
        self.SegmentCreatedArray = np.zeros(NumberOfSegments, dtype=np.float64)
        self.ArcCutoffPoint = np.zeros(NumberOfSegments, dtype=np.float64)
        self.RadiusOffsetArray = np.zeros(NumberOfSegments-1, dtype=np.float64)
        self.LeadInSegmentArray = np.zeros(NumberOfSegments-1, dtype=np.int)

    PrimarySegmentNumber = 0
    PrimarySegmentRadius = 0.0
    BlockSectionWidth = 0
    BlockSectionHeight = 0
    BlockSectionLength = 0
    EdgeStraightSectionLength = 0.0
    CenterBridgeLength = 0.0
    BlockSectionLength = 0.0
    ZSpatialPrecision = 0.0
    CalculatedPathDistance = 0.0
    SegmentLengthPrecision = 0.0
    NumberZSlices = 0

############################################################################################################
#
# WaveguidePropertiesDataObject
#
# Contains all information about the physical properties of the optical waveguide.  This information
# will be used to estimate light loss in the waveguide.  The parameters are based on Snyder and Love
# textbook and the calculations to determine these parameters are performed later on.

class WaveguidePropertiesDataObject(object):

##    def __init__(self, NumberOfSegments):
##        self.NumberOfSegments = NumberOfSegments

    # These parameters are set once and good for the physical properties of all waveguides
    IndexCore = 0.0
    IndexCladding = 0.0
    WaveguideRadius = 0.0        # The rho parameter
    Wavelength = 0.0            # microns
    NumericalAperture = 0.0
    NormalizedFrequency = 0.0   # The V parameter
    DeltaParameter = 0.0
    UParameter = 0.0
    WParameter = 0.0
    RadiusZero = 0.0            # The spot size - calculate on pg 341 using relationships there
    BetaCoefficient = 0.0
    

############################################################################################################
# ** AUXILIARY FUNCTIONS **
############################################################################################################
# Function - CalculateWaveguideProperties
#
# The parameters in the WaveguidePropertiesDataObject are calculated here.  The calculations are dervied
# from the textbook Optical Waveguide Theory by Snyder and Love, 1983.

def CalculateWaveguideProperties(WPData):

    WavePropData.IndexCore = 1.4927
    WavePropData.IndexCladding = 1.4877
    WavePropData.WaveguideRadius = 4.85
    WavePropData.Wavelength = 1.55  # microns

    # Numerical Aperture is based on the core and cladding indices
    WPData.NumericalAperture = np.sqrt(WPData.IndexCore**2 - WPData.IndexCladding**2)

    # Parameters from back of Snyder and Love
    WPData.NormalizedFrequency = ((2.*np.pi)/WPData.Wavelength) * WPData.WaveguideRadius * \
                                     WPData.NumericalAperture
    WPData.DeltaParameter = (WPData.IndexCore**2 - WPData.IndexCladding**2)/(2.*WPData.IndexCore**2)

    # Beta Coefficient is an arbitrary parameter based on the waveguide characteristics
##    BetaCoefficientArray = [6.041393049,6.03812908,6.035121234] from RSoft -> delta n = 0.005,0.004,0.003
    WPData.BetaCoefficient = 6.041393049

    # Given beta value, calculate W and U parameters
    kParam = (2.*np.pi)/WPData.Wavelength
    WPData.WParameter = WPData.WaveguideRadius * np.sqrt(WPData.BetaCoefficient**2 - \
                                                         (WPData.IndexCladding**2 * kParam**2))
    WPData.UParameter = WPData.WaveguideRadius * np.sqrt((WPData.IndexCore**2 * kParam**2) - \
                                                         WPData.BetaCoefficient**2)

    # Step Profile
    WPData.RadiusZero = WPData.WaveguideRadius/(np.sqrt(2. * np.log(WPData.NormalizedFrequency)))
    # Gaussian Profile
    WPData.RadiusZero = WPData.WaveguideRadius/np.sqrt(WPData.NormalizedFrequency - 1)

    print "*******************************************************************"
    print "WAVEGUIDE PROPERTIES"
    print "IndexCore: " + str(WPData.IndexCore)
    print "IndexCladding: " + str(WPData.IndexCladding)
    print "WaveguideRadius: " + str(WPData.WaveguideRadius)
    print "Wavelength: " + str(WPData.Wavelength)
    print "NumericalAperture: " + str(WPData.NumericalAperture)
    print "NormalizedFrequency: " + str(WPData.NormalizedFrequency)
    print "DeltaParameter: " + str(WPData.DeltaParameter)
    print "UParameter: " + str(WPData.UParameter)
    print "WParameter: " + str(WPData.WParameter)
    print "RadiusZero: " + str(WPData.RadiusZero)
    print "*******************************************************************"
    
    
############################################################################################################
# DESIGN FOR DETERMINING DISTANCES
############################################################################################################
# Function - CalculateInputOutputDistance
# This code calculates the differences in x and y between the input and output points of each spline
def CalculateInputOutputDistance(PDData):
    for i in range(0, len(PDData.XDistanceArray)):
        PDData.XDistanceArray[i] = PDData.PathXEndPosArray[i] - PDData.PathXStartPosArray[i] 
        PDData.YDistanceArray[i] = PDData.PathYEndPosArray[i] - PDData.PathYStartPosArray[i]
        #Direct distance formed by taking the hypotenuse of the X and Y difference above
        PDData.DirectDistanceArray[i] = np.sqrt((PDData.XDistanceArray[i])**2 + (PDData.YDistanceArray[i])**2)
    return

############################################################################################################
# Function - DeterminePrimarySegments
# Find which spline has the largest distance.  Returns the index of the largest segment

def DeterminePrimarySegments(DirectDistanceArray):
    LargestDistanceIndex = np.argmax(abs(DirectDistanceArray))
    print "Largest ordinal segment number: " + str(LargestDistanceIndex)
    return LargestDistanceIndex

############################################################################################################
# Function - DetermineSecondarySegments
# Find a list with the secondary segments

def DetermineSecondarySegments(PDData):
    PDData.SecondaryIndexList[0] = -1
    
    for i in range(0, PDData.NumberOfSegments):
        # If this segment doesn't exist in the largest list, add it to the secondary list
        if(i != PDData.PrimarySegmentNumber):
            #If the first item is -1 (first element in the list), replace, otherwise add
            if(PDData.SecondaryIndexList[0] == -1):
                PDData.SecondaryIndexList[0] = i
            else:
                PDData.SecondaryIndexList = np.append(PDData.SecondaryIndexList, i)

############################################################################################################
# AUX FUNCTIONS FOR BELOW
# Functions - For tridiagonal matrix deconvolving
# Adapted from Numerical Engineering Methods in Python (2005), Section 2.4

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b

############################################################################################################
# Function - CalculateSegmentLength
# Calculates the segment length based on the data points

def CalculateSegmentLength(PDData, XYZSegmentDataArray, SegNum):
    # Calculate spline length
    totalPathDistance = 0.0
    previousSliceXValue = PDData.PathXStartPosArray[SegNum]
    previousSliceYValue = PDData.PathYStartPosArray[SegNum]
    previousSliceZValue = 0.0
    
    for i in range(1, len(XYZSegmentDataArray[SegNum,:,0])):
        #Add the path amount to the total distance
        totalSliceDistance = np.sqrt((XYZSegmentDataArray[SegNum,i,0]-previousSliceXValue)**2 + \
                                     (XYZSegmentDataArray[SegNum,i,1]-previousSliceYValue)**2 + \
                                     (XYZSegmentDataArray[SegNum,i,2]-previousSliceZValue)**2)
        totalPathDistance += totalSliceDistance
        previousSliceXValue = XYZSegmentDataArray[SegNum,i,0]
        previousSliceYValue = XYZSegmentDataArray[SegNum,i,1]
        previousSliceZValue = XYZSegmentDataArray[SegNum,i,2]
    #end slice for loop

    return totalPathDistance

############################################################################################################
# DESIGN FOR CREATING SPLINE FOR EACH PATH
############################################################################################################
# Function - CreateSpline                    
# To calculate the path, the program uses the spline function.  This takes the starting and end
# points of the path, then returns the centerpoints of the function for each z-slice.
# http://www.physics.utah.edu/~detar/phys6720/handouts/cubic_spline/cubic_spline/node1.html
#
# This spline creation program uses the XY Distance plus the Z Distance to create the spline.  The
# XY difference will be the difference in YLineValues below on the line to pass in that will create
# it.  The Z difference will be used to create the XLineValues to pass in below.  The slopes of the
# line created will then be passed in to the code below.  This will return the 2nd derivative
# k values to pass into the Evaluate Spline function, which will return the distance values at each
# slice.  Using the X and Y beginning and end values of the segment will allow the spline to be
# translated into an array of X, Y values.

def CreateSpline(XYDistance, ZDistance, Invert = 0, DoubleSpline = 0, DoubleSplineOffset = 0.0):
    XLineValues = np.array([0.0,ZDistance/2.0,ZDistance]) # minimum of three points
    LineSlope = XYDistance/ZDistance
    YLineValues = LineSlope*XLineValues

    if (DoubleSpline == 1):
        YLineValues[1] += DoubleSplineOffset

    if (DoubleSpline == 2):
        XLineValues[1] = ZDistance/3.
        TwoThirdDataPointXVal = (ZDistance*2.)/3.
        YLineValues[1] = LineSlope * XLineValues[1]
        TwoThirdDataPointYVal = LineSlope * TwoThirdDataPointXVal

        YLineValues[1] += DoubleSplineOffset
        TwoThirdDataPointYVal -= DoubleSplineOffset
        
        XLineValues = np.insert(XLineValues, 2, TwoThirdDataPointXVal)
        YLineValues = np.insert(YLineValues, 2, TwoThirdDataPointYVal)
        
    if (DoubleSpline == 3):
        # For secondary spline, the line passed in needs to be adjusted by picking two more points at 1/4
        # and 3/4 of the total length and then incrementing these values vertically in the opposite
        # direction to create a 3 segment model.  This will increase the total value of the spline.
        # Increment the data points at 1/4 and 3/4 of the lengths to create the 3 segments
        OneQuarterDataPointXVal = ZDistance*.25
        ThreeQuarterDataPointXVal = ZDistance*.75
        OneQuarterDataPointYVal = LineSlope * OneQuarterDataPointXVal
        ThreeQuarterDataPointYVal = LineSlope * ThreeQuarterDataPointXVal

        # Increase these values by the offset
        OneQuarterDataPointYVal += DoubleSplineOffset
        ThreeQuarterDataPointYVal -= DoubleSplineOffset

        # Add these into the X and Y line value arrays
        XLineValues = np.insert(XLineValues, 1, OneQuarterDataPointXVal)
        XLineValues = np.insert(XLineValues, 3, ThreeQuarterDataPointXVal)
        YLineValues = np.insert(YLineValues, 1, OneQuarterDataPointYVal)
        YLineValues = np.insert(YLineValues, 3, ThreeQuarterDataPointYVal)
    
    x = XLineValues
    y = YLineValues

    # Slope specified at the left and right sides of the splines
    LeftTangent = 0.
    RightTangent = 0.
    
    # Mathematics from Numerical Methods using MATLAB 4th Edition, Section 5.3
    # with adapted code from Numerical Engineering Methods in Python (2005), Section 3.3
    # This code calculates the second derivative (k) at each knot specified
    n = len(x) - 1

    # The values for the clamped spline can be solved using equation 12 of the MATLAB
    # book.  This equation is only valid for the values from 1...n-1, with modifications for
    # 1 & n-1 made below.  To solve the equation, the values of k for 0 and n must be substituted

    # h is an intermediate variable that calculates the distance between x values
    h = np.zeros((n), dtype=np.float64)
    h[0:n] = x[1:n+1] - x[0:n]

    a = np.zeros((n), dtype=np.float64)
    b = np.ones((n+1), dtype=np.float64)
    c = np.zeros((n), dtype=np.float64)
    u = np.zeros((n+1), dtype=np.float64)
    k = np.zeros((n+1), dtype=np.float64)

    # Calculate matrix parameters: a + b + c = u
    a[0:n-1] = h[0:n-1]
    b[1:n] = 2.0*(h[0:n-1] + h[1:n])
    c[1:n] = h[1:n]
    u[1:n] = 6.0*(((y[2:n+1]-y[1:n])/h[1:n])-((y[1:n]-y[0:n-1])/h[0:n-1]))

    # The loop is only valid for parameters from 1 to n-1.  For the first value, the value for k[0]
    # is unknown, so the eqn. after the loop to calculate k[0] is substituted into the main loop eqn.,
    # solving for k[0]
    b[1] = 1.5*h[0] + 2*h[1]
    u[1] = u[1] - 3*(((y[1]-y[0])/h[0])-LeftTangent)
    
    # For the value of c, it depends on how many points the spline has.  If it is only three, there
    # is no need to proceed to the matrix solving loop, as there is only one parameter to solve for

    if(n == 2):
        # This means that the value k[2] is also unknown and calculated later.  There is also only
        # one unknown, k[1], so that can be solved now instead of needing matrix algebra
        k[1] = (u[1] - 3*(RightTangent - ((y[2]-y[1])/h[1])))/(1.5*(h[0]+h[1]))
    else:
        # There is more than one equation with more than one variable to be solved.
        # There is a special case if the loop variable is equal to n-1.  This means that the
        # value for k[n] is not known at this time, so a substitution will need to be performed
        # to be able to solve the matrix.  Otherwise the values are as above
        
        # There is no value for c (zero), b and u are modified
        b[n-1] = 2*h[n-2] + 1.5*h[n-1]
        u[n-1] = u[n-1] - 3*(RightTangent - ((y[n]-y[n-1])/h[n-1]))
        
        a,b,c = LUdecomp3(a,b,c)
        k = LUsolve3(a,b,c,u)

    # Plug in after solving for all center values
    k[0] = (3./h[0])*(((y[1]-y[0])/h[0]) - LeftTangent) - k[1]/2.
    k[n] = (3./h[n-1])*(RightTangent - ((y[n]-y[n-1])/h[n-1])) - k[n-1]/2.

    return x,y,k

############################################################################################################
# Function - EvaluateSpline
# Adapted from Numerical Engineering Methods in Python (2005)

def EvaluateSpline(XYKDataArray,x):

    xData = XYKDataArray[0]
    yData = XYKDataArray[1]
    kData = XYKDataArray[2]

    def findSegment(xData,x):
        iLeft = 0
        iRight = len(xData)- 1
        while 1:
            if (iRight-iLeft) <= 1: return iLeft
            i =(iLeft + iRight)/2
            if x < xData[i]: iRight = i
            else: iLeft = i
 
    i = findSegment(xData,x)
    h = xData[i] - xData[i+1]
    y = ((x - xData[i+1])**3/h - (x - xData[i+1])*h)*kData[i]/6.0 \
      - ((x - xData[i])**3/h - (x - xData[i])*h)*kData[i+1]/6.0   \
      + (yData[i]*(x - xData[i+1])                            \
       - yData[i+1]*(x - xData[i]))/h
    
    return y

############################################################################################################
# SPLINES
############################################################################################################
# Function DesignPrimarySpline

def DesignSplineSection(DirectDistance, ZDistance, ZPrecision, XStartPos, YStartPos, XDistance, YDistance, \
                        XYZSegDataArray, DoubleSpline = 0, DoubleSplineOffset = 0.0, AngleOffset = 0.0):

    # The code will create two splines, one for the x value in space and the other for the y value.
    # The code can be passed in an angle offset that will specify a direction to stretch the segment in.

    # The spline was created using the DirectDistanceArray.  This array falls on a line that lies
    # in the x-y plane and was created earlier using the starting and ending x-y points of the
    # segment.  We can use the slope of the line to determine the x and y coordinates of a point
    # that lies on this line.  The tangent of the corresponding angle equals the line slope

    XYLineAngle = np.arctan2(YDistance,XDistance)
    XDoubleSplineOffset = 0.
    YDoubleSplineOffest = 0.

    # All lateral stretching of the algorithm happens in the plane equal to the slope of the line
    # of the x-y projection of the start to end points of the line in the block.  To stretch the knot(s)
    # at another angle in space, an angle offset is specified here.  The angle is specified with
    # an angle of zero/2 pi (radians) in the x positive direction, although an angle offset of zero
    # means no offset.  Positive angle rotates in a clockwise direction looking from the start end toward
    # the end to match with the right hand curl rule.  The middle knot point is taken from the projection
    # line and then moved from the middle point of the segment in the angle specified.
    if (AngleOffset != 0.0):
        XDoubleSplineOffset = np.cos(AngleOffset)*abs(DoubleSplineOffset)
        YDoubleSplineOffset = np.sin(AngleOffset)*abs(DoubleSplineOffset)
        
    else:
        # Project along the original angle
        # X value = cos(angle)*hypotenuse, Y value = sin(angle)*hypotenuse
        XDoubleSplineOffset = np.cos(XYLineAngle)*DoubleSplineOffset
        # Need to correct for negative angles because the sine value is negative
        YDoubleSplineOffset = np.sin(XYLineAngle)*DoubleSplineOffset
    
    # Call the create spline routine to get the k (2nd derivative) values based on the lateral distance
    # and the length.
    XKSplineArray = np.array(CreateSpline(XDistance, ZDistance, DoubleSpline = DoubleSpline, \
                                           DoubleSplineOffset = XDoubleSplineOffset))
    YKSplineArray = np.array(CreateSpline(YDistance, ZDistance, DoubleSpline = DoubleSpline, \
                                           DoubleSplineOffset = YDoubleSplineOffset))
    
    # The k values are obtained for the x and y splines.  These values will be passed
    # into the Evaluate Spline function.  Now, the spline will be passed in x values to give the
    # interpolated y values at each slice.
    
    # Set the index of the current slice to zero
    sliceIndex = 0

    # Use to calculate each section and sum of the curve
    totalPathDistance = 0.0
    previousSliceXValue = 0.0
    previousSliceYValue = 0.0
    previousSliceZValue = 0.0
     
    for ZInterpolatedValue in range(0, int(ZDistance+ZPrecision), int(ZPrecision)):
        XInterpolatedValue = EvaluateSpline(XKSplineArray, ZInterpolatedValue)
        YInterpolatedValue = EvaluateSpline(YKSplineArray, ZInterpolatedValue)
        XYZSegDataArray[sliceIndex,2] = ZInterpolatedValue # Z value
        
        # Add the Interpolated Value to the start position
        NewXPosition = XStartPos + XInterpolatedValue
        NewYPosition = YStartPos + YInterpolatedValue
        XYZSegDataArray[sliceIndex,0] = NewXPosition
        XYZSegDataArray[sliceIndex,1] = NewYPosition

        #Add the path amount to the total distance
        totalSliceDistance = np.sqrt((XInterpolatedValue-previousSliceXValue)**2 + \
                                     (YInterpolatedValue-previousSliceYValue)**2 + \
                                     (ZInterpolatedValue-previousSliceZValue)**2)
        totalPathDistance += totalSliceDistance
        previousSliceXValue = XInterpolatedValue
        previousSliceYValue = YInterpolatedValue
        previousSliceZValue = ZInterpolatedValue
        sliceIndex += 1
    #end slice for loop
        
    return totalPathDistance

############################################################################################################
# Function - CreateSingleSpline
# This function goes through and creates the primary spline(s)
def CreateSingleSpline(PDData, segmentIndex, XYZSplineDataArray):

    XStartPos = PDData.PathXStartPosArray[segmentIndex]
    YStartPos = PDData.PathYStartPosArray[segmentIndex]
    XEndPos = PDData.PathXEndPosArray[segmentIndex]
    YEndPos = PDData.PathYEndPosArray[segmentIndex]
    XDistance = PDData.XDistanceArray[segmentIndex]
    YDistance = PDData.YDistanceArray[segmentIndex]
    DirectDistance = PDData.DirectDistanceArray[segmentIndex]

    print "DirectDistance: " + str(DirectDistance)
    print "XDistance: " + str(XDistance)
    print "XEndPos: " + str(XEndPos)
    
    NetSplineSectionLength = PDData.BlockSectionLength - (PDData.EdgeStraightSectionLength*2.0)
    
    # Create an XYZ Array for this segment index to pass into the design spline function.
    # It will be returned with data.
    SplineDataArray = np.zeros(((NetSplineSectionLength/PDData.ZSpatialPrecision)+1, 3), dtype=np.float64)

    # No primary conflict.  Pass in the distances to the spline creation algorithm
    SplineDistance = DesignSplineSection(DirectDistance, NetSplineSectionLength, PDData.ZSpatialPrecision, \
                            XStartPos, YStartPos, XDistance, YDistance, SplineDataArray)
    print "SingleSplineDistance: " + str(SplineDistance)

    # Now configure the data array by extrapolating out the end straight sections
    # Add the left section. X and Y values equal to the starting position.  Fill in Z values.
    FirstSectionJoinIndex = int(PDData.EdgeStraightSectionLength/PDData.ZSpatialPrecision)
    XYZSplineDataArray[0:FirstSectionJoinIndex,0] = XStartPos
    XYZSplineDataArray[0:FirstSectionJoinIndex,1] = YStartPos
    ZIncrementArray = np.arange(0.0,PDData.EdgeStraightSectionLength, PDData.ZSpatialPrecision, dtype=np.float64)
    XYZSplineDataArray[0:FirstSectionJoinIndex,2] = ZIncrementArray

    # Copy the center section.  Increment the z-values to match up with beginning straight section
    SplineDataArray[:,2] += PDData.EdgeStraightSectionLength
    SecondSectionJoinIndex = int((PDData.EdgeStraightSectionLength + NetSplineSectionLength) \
                                     /PDData.ZSpatialPrecision)+1
    XYZSplineDataArray[FirstSectionJoinIndex:SecondSectionJoinIndex,:] = SplineDataArray

    #Add in the end section
    XYZSplineDataArray[SecondSectionJoinIndex:,2] = ZIncrementArray + (PDData.EdgeStraightSectionLength + \
                                                            NetSplineSectionLength + PDData.ZSpatialPrecision)
    XYZSplineDataArray[SecondSectionJoinIndex:,0] = XEndPos
    XYZSplineDataArray[SecondSectionJoinIndex:,1] = YEndPos

    # Return the total distance
    return SplineDistance

############################################################################################################
# Function - CreatePrimarySegments
# This function goes through and creates the primary spline(s)

def CreatePrimarySegments(PDData, XYZSegmentDataArray):

    NoPrimarySegmentError = 1
    # First calculate the primary ones

    SegmentIndex = PDData.PrimarySegmentNumber
    XYZSplineDataArray = np.zeros(((PDData.BlockSectionLength/PDData.ZSpatialPrecision)+1, 3), \
                                          dtype=np.float64)

    SplineDistance = CreateSingleSpline(PDData, SegmentIndex, XYZSplineDataArray)

    if (SplineDistance > 0):
        XYZSegmentDataArray[SegmentIndex,:,:] = XYZSplineDataArray
        PDData.SegmentCreatedArray[SegmentIndex] = 1

    #CalculatedPathDistance is an array
    PDData.CalculatedPathDistance = SplineDistance + (PDData.EdgeStraightSectionLength*2)
    print "Segment distance: " + str(PDData.CalculatedPathDistance)

    return NoPrimarySegmentError

############################################################################################################
# Function - CreateDoubleSpline
# This function goes through and creates the primary spline(s)

def CreateDoubleSpline(PDData, segmentIndex, XYZSplineDataArray, DoubleSpline = 1, AngleOffset = 0.0, \
                           SegmentLead = 0):

    NetSplineSectionLength = PDData.BlockSectionLength - (PDData.EdgeStraightSectionLength*2.0) - \
                                 (SegmentLead*PDData.ZSpatialPrecision)
    
    # Create an XYZ Array for this segment index to pass into the design spline function.
    # It will be returned with data.
    SplineDataArray = np.zeros(((NetSplineSectionLength/PDData.ZSpatialPrecision)+1, 3), dtype=np.float64)

    XStartPos = PDData.PathXStartPosArray[segmentIndex]
    YStartPos = PDData.PathYStartPosArray[segmentIndex]
    XEndPos = PDData.PathXEndPosArray[segmentIndex]
    YEndPos = PDData.PathYEndPosArray[segmentIndex]
    XDistance = PDData.XDistanceArray[segmentIndex]
    YDistance = PDData.YDistanceArray[segmentIndex]
    DirectDistance = PDData.DirectDistanceArray[segmentIndex]
    ZSpatialPrecision = PDData.ZSpatialPrecision
    EdgeStraightSectionLength = PDData.EdgeStraightSectionLength

    print "##################"
    XMidPos = 0.
    YMidPos = 0.
    XDifference = XEndPos - XStartPos
    YDifference = YEndPos - YStartPos

    print "Start/End Positions"
##    print XStartPos, YStartPos
##    print XEndPos, YEndPos

    XYLineAngle = np.arctan2(YDistance, XDistance)

    print "LineAngle: " + str(XYLineAngle)

    # This while loop will repeat the code below until the spline length comes within the tolerance specified
    IncrementValue = 5.0
    SegmentDistanceDifference = 0.0
    LastSegmentDistanceDifference = 0.0
    ToleranceAchieved = 0
    RunNumber = 1
    SwitchToFlipMethod = 0
    OffsetValue = 5.0

    print "Angle offset: " + str(AngleOffset)
    
    while(ToleranceAchieved == 0):
        
        SplineDistance = DesignSplineSection(DirectDistance, NetSplineSectionLength, PDData.ZSpatialPrecision, \
                                XStartPos, YStartPos, XDistance, YDistance, SplineDataArray, \
                                DoubleSpline = DoubleSpline, DoubleSplineOffset = OffsetValue, \
                                AngleOffset = AngleOffset)
        
        # Now configure the data array by extrapolating out the end straight sections
        # Add the left section. X and Y values equal to the starting position.  Fill in Z values.
        FirstSectionJoinIndex = int(EdgeStraightSectionLength/ZSpatialPrecision + SegmentLead)
        XYZSplineDataArray[0:FirstSectionJoinIndex,0] = XStartPos
        XYZSplineDataArray[0:FirstSectionJoinIndex,1] = YStartPos
        ZIncrementArray = np.arange(0.0,EdgeStraightSectionLength+(SegmentLead*ZSpatialPrecision), \
                                        ZSpatialPrecision, dtype=np.float64)
        XYZSplineDataArray[0:FirstSectionJoinIndex,2] = ZIncrementArray

        # Copy the center section.  Increment the z-values to match up with beginning straight section
        SplineDataArray[:,2] += EdgeStraightSectionLength + SegmentLead*ZSpatialPrecision
        SecondSectionJoinIndex = int((EdgeStraightSectionLength + NetSplineSectionLength)/ZSpatialPrecision) + \
                                         SegmentLead + 1
        XYZSplineDataArray[FirstSectionJoinIndex:SecondSectionJoinIndex,:] = SplineDataArray

        #Add in the end section
        ZIncrementArray2 = np.arange(0.0,EdgeStraightSectionLength, ZSpatialPrecision, dtype=np.float64)
        XYZSplineDataArray[SecondSectionJoinIndex:,2] = ZIncrementArray2 + (EdgeStraightSectionLength + \
                                                                NetSplineSectionLength + \
                                                                (1+SegmentLead)*ZSpatialPrecision)
        XYZSplineDataArray[SecondSectionJoinIndex:,0] = XEndPos
        XYZSplineDataArray[SecondSectionJoinIndex:,1] = YEndPos


        # Now see how the returned value compares to the needed spline distance
        NewSegmentDistance = 2*EdgeStraightSectionLength + SegmentLead*ZSpatialPrecision + SplineDistance
        SegmentDistanceDifference = PDData.CalculatedPathDistance - NewSegmentDistance
        if(abs(SegmentDistanceDifference) <= PDData.SegmentLengthPrecision):
            # The segment is within tolerance.  Use these values and exit the function
            ToleranceAchieved = 1
        else:
            ThisRun = np.sign(SegmentDistanceDifference)
            LastRun = np.sign(LastSegmentDistanceDifference)
            
            if ((ThisRun != LastRun and LastSegmentDistanceDifference != 0.0) or SwitchToFlipMethod == 1):
                # The segment distance is more than the needed distance, but not within tolerance
                # Flip the sign and reduce the amount by 5%
                SwitchToFlipMethod = 1
                IncrementValue = (-0.95) * IncrementValue

            # Now increase the midpoint           
            OffsetValue += IncrementValue
            #Update the last run value
            LastSegmentDistanceDifference = SegmentDistanceDifference

        RunNumber += 1
    #end while
    
    print "Run number: " + str(RunNumber)
    print "Offset Value: " + str(OffsetValue)
    print "SplineDistance: " + str(SplineDistance)

    # Return the total spline distances
    return NewSegmentDistance

############################################################################################################
# Function - CreateSecondarySegments
# This function goes through and creates the secondary spline(s)

def CreateSecondarySegments(PDData, XYZSegmentDataArray):

    #Go through the list of secondary segments and check for any conflicts
    ConflictFlag = 0
    PrimaryConflictFlag = 0

    # KEY 0-red, 1-green, 2-orange, 3-blue, 4-pink, 5-gold, 6-aqua, 7-yellow, 8-white
    # Array should have one less value than total segments, as primary is already created
    AngleOffsetArray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    DoubleSplineArray = np.array([1,1,1,1,1,1,1])
    LeadInSegmentArray = np.array([0, 0, 0, 0, 0, 0, 0])

    print "Secondary Segment List: " + str(PDData.SecondaryIndexList)

    for i in range(0, len(PDData.SecondaryIndexList)):
    
        SegmentIndex = PDData.SecondaryIndexList[i]

        XYZSplineDataArray = np.zeros(((PDData.BlockSectionLength/PDData.ZSpatialPrecision)+1, 3), \
                                          dtype=np.float64)

        AngleOffset = AngleOffsetArray[i]
        DoubleSpline = DoubleSplineArray[i]
        SegmentLead = LeadInSegmentArray[i]

        SplineDistance = CreateDoubleSpline(PDData, SegmentIndex, XYZSplineDataArray, \
                           DoubleSpline = DoubleSpline, AngleOffset = AngleOffset, \
                           SegmentLead = SegmentLead)

        TotalDistance = SplineDistance
        print "Segment " + str(SegmentIndex) + " has a distance of " + str(TotalDistance)

        if (SplineDistance > 0):
            XYZSegmentDataArray[SegmentIndex,:,:] = XYZSplineDataArray
            PDData.SegmentCreatedArray[SegmentIndex] = 1

############################################################################################################
# CIRCULAR ARCS
############################################################################################################
# Function - CreatePrimaryArcSegment                  
# Function to create the primary arc segment

def CreatePrimaryArcSegment(PDData, XYZSegmentDataArray, ArcRadiusDataArray):
    CreateDoubleArcSegment(PDData, XYZSegmentDataArray, PDData.PrimarySegmentNumber)
    ArcRadiusDataArray[PDData.PrimarySegmentNumber] = PDData.PrimarySegmentRadius

############################################################################################################
# Function - CreateDoubleArcSegment                  
# This function is intended to create a particular segment by using two or three circular arcs connected
# at the intersection of each arc.  Setting parameter MatchLength equal to 1 means the algorithm will
# run recursively until the segment length is matched.

def CreateDoubleArcSegment(PDData, XYZSegmentDataArray, SegNum):

    # The mathematical basis for this function is to minimize the curvature of the segments by representing
    # them as the interconnection of two circular arcs for a basic spline, or if more distance is needed
    # to use three circular arcs.  The math function will be:
    # Radius1*Angle1 + Radius2*Angle2 = Segment Length
   
    Width = PDData.DirectDistanceArray[SegNum]
    Length = PDData.BlockSectionLength - 2.*PDData.EdgeStraightSectionLength

    # Basic arc would be two circles with identical radius and arc angle
    # Chord length is the hypotenuse formed by the width and length (divided by 2) which is used as
    # the basis for the radius and angle of the arc
    ChordLength = np.sqrt(Length**2 + Width**2)/2.
    # Angle is formed by the length and width and forms the other angle of the triangle
    # formed by the bisector of the chord
    ChordAngle = np.arctan2(Length,Width)
    # The angle and radius of the arc
    ArcAngle = 2.*(np.pi/2. - abs(ChordAngle))
    ArcRadius = (ChordLength/2.)/np.cos(ChordAngle)

    print "*******************************************************************"
    print "Primary Segment: " + str(SegNum)
    print "Width of segment is: " + str(Width)
    
    # Now with this information, we can construct the circle and interpolate points along the arc
    # X below is the single lateral dimension of the arc, which will be translated into X and Y using
    # the angle of x and y formed by the start and end points and Y below is the length or Z dimension
    LeftArcCenterPointX = 0
    LeftArcCenterPointY = 1.*ArcRadius
    XDelta = PDData.ZSpatialPrecision
    XSlices = int(Length/XDelta)
    ArcSegmentDataArray = np.zeros((XSlices+1, 2), dtype=np.float64)
    # The left arc will take care of the arc inflection point at the closest slice
    LeftXArcWidth = np.sin(ChordAngle) * ChordLength
    LeftXSlice = LeftXArcWidth/XDelta
    # The left arc will take care of the center, arc inflection point near halfway
    # Now do a mirror image for the Right Arc with the remaining length
    RightArcCenterPointY = -1.*(ArcRadius-Width)

    for i in range(0, XSlices+1):
        XValue = i*XDelta

        if(i <= LeftXSlice):
            # Equation to find y value is based on x = r*cos(theta) and y = r*sin(theta)
            ThetaValue = np.arcsin(XValue/ArcRadius)
            YValue = LeftArcCenterPointY - (ArcRadius*np.cos(ThetaValue))
        else:
            ThetaValue = np.arcsin((Length - XValue)/ArcRadius)
            YValue = RightArcCenterPointY + (ArcRadius*np.cos(ThetaValue))
        
        ArcSegmentDataArray[i,0] = XValue
        ArcSegmentDataArray[i,1] = YValue
        
    #####
    # Now interpolate this curve into X and Y dimensions based on the original line slope
    XYLineAngle = np.arctan2(PDData.XDistanceArray[SegNum],PDData.YDistanceArray[SegNum])
    XStartPos = PDData.PathXStartPosArray[SegNum]
    YStartPos = PDData.PathYStartPosArray[SegNum]
    XEndPos = PDData.PathXEndPosArray[SegNum]
    YEndPos = PDData.PathYEndPosArray[SegNum]

    FirstSectionJoinIndex = int(PDData.EdgeStraightSectionLength/PDData.ZSpatialPrecision)
    XYZSegmentDataArray[SegNum,0:FirstSectionJoinIndex,0] = XStartPos
    XYZSegmentDataArray[SegNum,0:FirstSectionJoinIndex,1] = YStartPos
    ZIncrementArray = np.arange(0.0,PDData.EdgeStraightSectionLength, PDData.ZSpatialPrecision, dtype=np.float64)
    XYZSegmentDataArray[SegNum,0:FirstSectionJoinIndex,2] = ZIncrementArray

    # Loop through ArcSegmentDataArray to extrapolate segment into X and Y values
    for i in range(0, len(ArcSegmentDataArray[:,0])):
        # The Z value equals the x value for the arc segment data array
        XYZSegmentDataArray[SegNum,i+FirstSectionJoinIndex,2] = ArcSegmentDataArray[i,0] + \
                                                                PDData.EdgeStraightSectionLength
        # Increment the values by X and Y based on line angle
        XYZSegmentDataArray[SegNum,i+FirstSectionJoinIndex,0] = XStartPos + \
                                            (ArcSegmentDataArray[i,1]*np.sin(XYLineAngle))
        XYZSegmentDataArray[SegNum,i+FirstSectionJoinIndex,1] = YStartPos + \
                                            (ArcSegmentDataArray[i,1]*np.cos(XYLineAngle))

    #Add in the end section
    SecondSectionJoinIndex = int((PDData.EdgeStraightSectionLength + Length)/PDData.ZSpatialPrecision)+1
    XYZSegmentDataArray[SegNum,SecondSectionJoinIndex:,2] = ZIncrementArray + \
                                            (PDData.EdgeStraightSectionLength + \
                                            Length + PDData.ZSpatialPrecision)
    XYZSegmentDataArray[SegNum,SecondSectionJoinIndex:,0] = XEndPos
    XYZSegmentDataArray[SegNum,SecondSectionJoinIndex:,1] = YEndPos
    #####

    totalPathDistance = CalculateSegmentLength(PDData, XYZSegmentDataArray, SegNum)
    print "Segment length: " + str(totalPathDistance)

    # Primary segment
    PDData.CalculatedPathDistance = totalPathDistance
    PDData.PrimarySegmentRadius = ArcRadius
    print "Radius length: " + str(ArcRadius)
           
############################################################################################################
# Function - CreateSecondaryArcSegments                  
# Function to create the secondary waveguides.  Breaks the 3D waveguide creation up into x prime and y prime
# arcs.  These are interpolated to form a 3D waveguide with respect to the x,y prime axis.  This is then
# translated to x and y based on the radius offset that the endpoints site with respect to the x,y axis.

def CreateSecondaryArcSegments(PDData, XYZSegmentDataArray, ArcRadiusDataArray):

    # This scaling factor works as follows: The x dimension uses the current radius, starting with the
    # radius of the primary waveguide.  The current radius is then multiplied by the scaling factor to
    # give the radius to be used in the y dimension.  The respective triple arc waveguides are created
    # using those radii and then interpolated to give the three dimensional waveguide.  A larger number
    # will scale the y dimension less, whereas a value of 1 should give a 45 degree "rotation".  A
    # negative scaling factor will move the y-value in the negative direction.  If zero, there will
    # be no additional y value

    # KEY 0-red, 1-green, 2-orange, 3-blue, 4-pink, 5-gold, 6-aqua, 7-yellow, 8-white
    RadiusOffsetArray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    LeadInSegmentArray = np.array([0, 0, 0, 0, 0, 0, 0])
    ReverseXOn = 1

    print "Secondary Segment List: " + str(PDData.SecondaryIndexList)
    for i in range(0, len(PDData.SecondaryIndexList)):
        SegmentIndex = PDData.SecondaryIndexList[i]
        SegmentLead = LeadInSegmentArray[i]
        NetArcSectionLength = PDData.BlockSectionLength - (PDData.EdgeStraightSectionLength*2.0) - \
                              SegmentLead * PDData.ZSpatialPrecision
        XArcDataArray = np.zeros(((NetArcSectionLength/PDData.ZSpatialPrecision)+1, 2), \
                                          dtype=np.float64)
        YArcDataArray = np.zeros(((NetArcSectionLength/PDData.ZSpatialPrecision)+1, 1), \
                                          dtype=np.float64)
        TmpXArcDataArray = np.zeros(((NetArcSectionLength/PDData.ZSpatialPrecision)+1, 1), \
                                          dtype=np.float64)
        PreAdjustYArcDataArray = np.zeros(((NetArcSectionLength/PDData.ZSpatialPrecision)+1, 2), \
                                          dtype=np.float64)
        RadiusOffset = RadiusOffsetArray[i]

        # Angle to scale into x and y
        XYLineAngle = np.arctan2(PDData.XDistanceArray[SegmentIndex],PDData.YDistanceArray[SegmentIndex])
        XStartPos = PDData.PathXStartPosArray[SegmentIndex]
        YStartPos = PDData.PathYStartPosArray[SegmentIndex]
        XEndPos = PDData.PathXEndPosArray[SegmentIndex]
        YEndPos = PDData.PathYEndPosArray[SegmentIndex]
        XWidth = PDData.XDistanceArray[SegmentIndex]
        YWidth = PDData.YDistanceArray[SegmentIndex]
        Length = PDData.BlockSectionLength - 2.*PDData.EdgeStraightSectionLength

        print "*******************************************************************"
        print "Segment: " + str(SegmentIndex)

        # The Radius here is based on the primary radius calculated for the primary arc segment
        # The radius (for all 3 circles) will be incrementally decreased, which increases the length of the
        # segment.  When the segment length is achieved, the loop exits
        CurrentRadius = PDData.PrimarySegmentRadius
        OffsetIncrement = CurrentRadius/1000.
        CurrentRadius -= OffsetIncrement
        CurrentYRadius = abs(CurrentRadius * RadiusOffset)
        DistanceMatched = 0
        SegmentDistanceDifference = 0.0
        LastSliceValue = 0
        LastSegmentDistanceDifference = 0.0
        RunNumber = 1
        SwitchToFineMethod = 0
        PreviousCntr2Slice = 0 #TEMP
        
        while(DistanceMatched == 0):

            # Create the arc based waveguides for x and y.
            CreateTripleArcSegment(CurrentRadius, XArcDataArray, XWidth, NetArcSectionLength, \
                                   PDData.ZSpatialPrecision)
            CreateTripleArcSegment(CurrentYRadius, PreAdjustYArcDataArray, abs(YWidth), NetArcSectionLength, \
                                   PDData.ZSpatialPrecision)

            # The y data needs to be scaled based on the value of the width, as well as the y radius scaling factor
            if(YWidth > 0.0 and RadiusOffset < 0.0):
                # Flip the array data left to right
                YArcDataArray = np.copy(PreAdjustYArcDataArray[:,1])
                YArcDataArray = YArcDataArray[::-1]
                # Add the width
                YArcDataArray -= YWidth
                # Now flip top to bottom
                YArcDataArray = YArcDataArray[:] * -1.0
                YArcDataArray = YArcDataArray.reshape(len(YArcDataArray[:]),1)
            elif(YWidth < 0.0 and RadiusOffset > 0.0):
                # Flip the array data left to right
                YArcDataArray = np.copy(PreAdjustYArcDataArray[:,1])
                YArcDataArray = YArcDataArray[::-1]
                # Add the negative width
                YArcDataArray += YWidth
                YArcDataArray = YArcDataArray.reshape(len(YArcDataArray[:]),1)
            elif(YWidth < 0.0 and RadiusOffset <= 0.0):
                # Flip the array data top to bottom
                YArcDataArray = np.copy(PreAdjustYArcDataArray[:,1])
                YArcDataArray = YArcDataArray[:] * -1.0
                YArcDataArray = YArcDataArray.reshape(len(YArcDataArray[:]),1)
            else:
                YArcDataArray = np.copy(PreAdjustYArcDataArray[:,1])

            # FLIP X FOR REVERSE BEND
            if(ReverseXOn == 1):
                TmpXArcDataArray = np.copy(XArcDataArray[:,1])
                # Flip the array data left to right
                TmpXArcDataArray = TmpXArcDataArray[::-1]
                # Now flip top to bottom
                TmpXArcDataArray = TmpXArcDataArray[:] * -1.0
                # Add the width
                TmpXArcDataArray += XWidth
                XArcDataArray[:,1] = TmpXArcDataArray

            # Move the segment interpolation to this section, along with the creation of a triple arc
            # x, triple arc y curve.  The portion before this should consist of using the ROC from
            # the primary arc, and adjusting that in each triple arc creation section.

            # Now interpolate this curve into X and Y dimensions based on the original line slope
            FirstSectionJoinIndex = int(PDData.EdgeStraightSectionLength/PDData.ZSpatialPrecision + \
                                        SegmentLead)
            XYZSegmentDataArray[SegmentIndex,0:FirstSectionJoinIndex,0] = XStartPos
            XYZSegmentDataArray[SegmentIndex,0:FirstSectionJoinIndex,1] = YStartPos
            ZIncrementArray = np.arange(0.0,PDData.EdgeStraightSectionLength+(SegmentLead * \
                                    PDData.ZSpatialPrecision), PDData.ZSpatialPrecision, dtype=np.float64)
            XYZSegmentDataArray[SegmentIndex,0:FirstSectionJoinIndex,2] = ZIncrementArray

            # Loop through ArcSegmentDataArray to extrapolate segment into X and Y values
            for i in range(0, len(XArcDataArray[:,0])):
                # The Z value equals the x value for the arc segment data array
                XYZSegmentDataArray[SegmentIndex,i+FirstSectionJoinIndex,2] = XArcDataArray[i,0] + \
                                        PDData.EdgeStraightSectionLength + \
                                        (SegmentLead * PDData.ZSpatialPrecision)
                # Now insert the x and y waveguide data
                XYZSegmentDataArray[SegmentIndex,i+FirstSectionJoinIndex,0] = XStartPos + XArcDataArray[i,1]
                XYZSegmentDataArray[SegmentIndex,i+FirstSectionJoinIndex,1] = YStartPos + YArcDataArray[i]

            #Add in the end section
            SecondSectionJoinIndex = int((PDData.EdgeStraightSectionLength + NetArcSectionLength) \
                                         /PDData.ZSpatialPrecision) + SegmentLead + 1
            ZIncrementArray2 = np.arange(0.0,PDData.EdgeStraightSectionLength, PDData.ZSpatialPrecision, \
                                         dtype=np.float64)
            XYZSegmentDataArray[SegmentIndex,SecondSectionJoinIndex:,2] = ZIncrementArray2+ \
                            (PDData.EdgeStraightSectionLength + NetArcSectionLength + \
                            (1+SegmentLead)*PDData.ZSpatialPrecision)
            XYZSegmentDataArray[SegmentIndex,SecondSectionJoinIndex:,0] = XEndPos
            XYZSegmentDataArray[SegmentIndex,SecondSectionJoinIndex:,1] = YEndPos
            #####

            totalPathDistance = CalculateSegmentLength(PDData, XYZSegmentDataArray, SegmentIndex)

            # Determine if path length is within tolerance
            SegmentDistanceDifference = PDData.CalculatedPathDistance - totalPathDistance
            if(abs(SegmentDistanceDifference) <= PDData.SegmentLengthPrecision):
                DistanceMatched = 1
            else:
                ThisRun = np.sign(SegmentDistanceDifference)
                LastRun = np.sign(LastSegmentDistanceDifference)

                if ((ThisRun != LastRun and LastSegmentDistanceDifference != 0.0) or SwitchToFineMethod == 1):
                    # The segment distance is more than the needed distance, but not within tolerance
                    # To fix this we go back by twice the amount of the offset increment and then adjust
                    # the radius at an offset of one tenth the amount of the previous offset
                    if(SwitchToFineMethod == 0):
                        print "FINE ADJUSTMENT"
                        CurrentRadius += 2*OffsetIncrement
                        OffsetIncrement = OffsetIncrement*0.1
                    SwitchToFineMethod = 1

                #Update the last run value
                LastSegmentDistanceDifference = SegmentDistanceDifference
                CurrentRadius -= OffsetIncrement
                CurrentYRadius = abs(CurrentRadius * RadiusOffset)

            RunNumber += 1

##        print CurrentRadius
##        print CurrentYRadius

        ArcRadiusDataArray[SegmentIndex] = CurrentRadius

        print SegmentLead, NetArcSectionLength

        print "Radius length: " + str(CurrentRadius)
        print "Total runs: " + str(RunNumber)

############################################################################################################
# Function - CreateTripleArcSegment                  
# This function is intended to create a particular segment by using three circular arcs connected at the 
# intersection of each arc.  In addition it uses an additional angle to rotate the shapes in space.

def CreateTripleArcSegment(CurrentRadius, ArcDataArray, Width, Length, XDelta):

    # The segment here is created using three arcs based on circles of identical radii.  Looking from
    # the side, the first arc will be constructed from the bottom-right portion of circle 1, with
    # the bottom at the input, the third arc will be constructed from the bottom-left portion of circle
    # 3, with the remaining portion of the segment being created by the top part of circle 2, with
    # the arc going from where the circle touches the other two.
    # The math function will be:
    # Radius1*Angle1 + Radius2*Angle2 + Radius3*Angle3 = Segment Length
    # The centers of circles 1 through 3 all have an X and Y coordinate
    Center1X = 0.0
    Center3X = Length

    # If the current radius is set to zero, calculate what radius will make for a simple double arc curve
    if(CurrentRadius == 0.0):
        ChordLength = np.sqrt(Length**2 + Width**2)/2.
        ChordAngle = np.arctan2(Length,Width)
        CurrentRadius = (ChordLength/2.)/np.cos(ChordAngle)

    # First calculate the locations of circles 1 and 3.  Start point on circle 1 is 0,0
    Center1Y = CurrentRadius
    Center3Y = CurrentRadius + Width
    # This is the distance between the two centers of circles 1 and 3
    DistanceCenters13 = np.sqrt((Center3Y-Center1Y)**2 + (Center3X-Center1X)**2)
    # Angles 1-3 are the arc angles that will be used for circles 1-3 in the segment
    # Angle 4 is calculated by using the bisect of the Distance of the centers
    # Angle 4 is for both arcs, as the bisect creates an isoceles triangle
    Angle4 = np.arccos((DistanceCenters13/2)/(2*CurrentRadius))
    Angle1 = np.pi/2. + np.arcsin(abs(Center3Y-Center1Y)/DistanceCenters13) - Angle4
    Angle3 = np.arccos(abs(Center3Y-Center1Y)/DistanceCenters13) - Angle4
    Center2X = 2*CurrentRadius*np.sin(Angle1) + Center1X
    Center2Y = -2*CurrentRadius*np.cos(Angle1) + Center1Y

    # Intersection point 1 is where arc 1 meets arc 2, point 2 is where arc 2 meets arc 3
    IntersectPoint1X = CurrentRadius*np.sin(Angle1)
    IntersectPoint1Y = Center1Y - CurrentRadius*np.cos(Angle1)
    IntersectPoint2X = Length - CurrentRadius*np.sin(Angle3)
    IntersectPoint2Y = Center3Y - CurrentRadius*np.cos(Angle3)

    # Now construct the segment starting from Arc 1 and build the data array.
    XSlices = int(Length/XDelta)

    # The intersect points may or may not coincide with an x slice value.  This is OK, but
    # the surrounding slices need to be identified as to which arc to use for each
    Arc1Length = np.sin(Angle1) * CurrentRadius
    Arc1XSlice = Arc1Length/XDelta
    
    # Arc2 X Slices go from Arc1 Slices + 1 to Arc2 X Slices.  Arc 2 end point is at Arc 3
    Arc3Length = np.sin(Angle3) * CurrentRadius
    Arc2XSlice = XSlices - Arc3Length/XDelta

##    print Arc1XSlice, Arc2XSlice
        
    for i in range(0, XSlices+1):
        XValue = i*XDelta
        if(i <= Arc1XSlice):
            # Equation to find y value is based on x = r*cos(theta) and y = r*sin(theta)
            ThetaValue = np.arcsin(XValue/CurrentRadius)
            YValue = Center1Y - (CurrentRadius*np.cos(ThetaValue))
        elif(i > Arc1XSlice and i <= Arc2XSlice):
            # Use arc 2
            ThetaValue = np.arcsin(abs(Center2X-(i*XDelta))/CurrentRadius)
            YValue = Center2Y + (CurrentRadius*np.cos(ThetaValue))
        else:
            # Use arc 3
            ThetaValue = np.arcsin((Length-XValue)/CurrentRadius)
            YValue = Center3Y - (CurrentRadius*np.cos(ThetaValue))
        
        ArcDataArray[i,0] = XValue
        ArcDataArray[i,1] = YValue
        
############################################################################################################        
############################################################################################################
# FINAL CHECK

def SegmentDistanceCurvatureCheck(XYZSegmentDataArray,PDData, WPData, ArcRadiusDataArray):

    print "***** CURVATURES *****"
    SegmentCreatedArray = PDData.SegmentCreatedArray
    MinimumSeparationDistance = PDData.MinimumPathSeparationDist
    ZSpatialPrecision = PDData.ZSpatialPrecision
    PrimarySegmentNumber = PDData.PrimarySegmentNumber
    PrimaryWaveguideLength = PDData.CalculatedPathDistance
    
    # Loop through each segment and at each point in the segment, compare the separation distance
    # to each of the other segments.
    ClosestDistance = 10000.
    NoSegmentConflictFound = 1
    MinimumYDistance = 10000.
    MaximumYDistance = 0.
    MinYDistZValue = 0.
    MaxYDistZValue = 0.
    MinimumYDistSegment = 0
    MaximumYDistSegment = 0
    ClosestSegment1 = 0
    ClosestSegment2 = 0
    ClosestZValue = 0.
    NumberOfSegments = len(SegmentCreatedArray)
    NumPoints = len(XYZSegmentDataArray[0,:,0])
    IsPrimary = 1
    
    # Outer loop is the segment to check
    for i in range(0,NumberOfSegments):
        #Middle loop is each segment to verify against.  Need a flag variable to exit loop if conflict found
        print "************************************************************"
        print "CURVATURE FOR WAVEGUIDE " + str(i)
        
        for j in range(0,NumberOfSegments):
            if(i != j):
                # Not the same segment, so step through each point in the array.  Inner loop is the z values
                for k in range(0,len(XYZSegmentDataArray[0,:,0])):
                    XDiff = XYZSegmentDataArray[i,k,0] - XYZSegmentDataArray[j,k,0]
                    YDiff = XYZSegmentDataArray[i,k,1] - XYZSegmentDataArray[j,k,1]
                    DirectDifference = np.sqrt((XDiff)**2 + (YDiff)**2)

                    # Also want to record the closest distance between any two lines
                    if(DirectDifference < ClosestDistance):
                        ClosestDistance = DirectDifference
                        ClosestSegment1 = i
                        ClosestSegment2 = j
                        ClosestZValue = XYZSegmentDataArray[j,k,2]

                    # Find the minimum and maximum y values for laser burn distance
                    if(XYZSegmentDataArray[i,k,1] < MinimumYDistance):
                        MinimumYDistance = XYZSegmentDataArray[i,k,1]
                        MinimumYDistSegment = i
                        MinYDistZValue = XYZSegmentDataArray[j,k,2]
                    if(XYZSegmentDataArray[i,k,1] > MaximumYDistance):
                        MaximumYDistance = XYZSegmentDataArray[i,k,1]
                        MaximumYDistSegment = i
                        MaxYDistZValue = XYZSegmentDataArray[j,k,2]
      
        # Also need to calculate the radius of curvature.  Take the difference in x and y and calculate
        # it as a radial difference versus the distance in z.  This gives an angle in cylindrical coordinates
        # versus z.  The arctan of that gives the angle.  Once the angle is calculated, take the difference
        # in angle versus the difference in length of the segment for the radius of curvature.
        MinimumROC = 1.0e30
        MaximumCurvature = 0.
        ROCZValue = 0
        CurvatureZValue = 0

        SlopeAngleArray = np.zeros(NumPoints, dtype=np.float64)
        SlopeAngleArray[0] = 0  # due to horizontal input
        CurvatureArray = np.zeros(NumPoints, dtype=np.float64)
        AngleCurvatureArray = np.zeros(NumPoints, dtype=np.float64)
        RadiusOfCurvatureArray = np.zeros(NumPoints, dtype=np.float64)
        SegmentDistanceArray = np.zeros(NumPoints+1, dtype=np.float64)
        XDifferenceArray = np.zeros(NumPoints+1, dtype=np.float64)

        # Calculate the segment distances for the entire array
        # End segment distances are arbitrarily set to a value to calculate the angle difference
        ArbitraryEndSegLength = 25.
        SegmentDistanceArray[0] = ArbitraryEndSegLength
        SegmentDistanceArray[NumPoints] = ArbitraryEndSegLength

        for m in range(1, NumPoints):
            SegmentDistanceArray[m] = \
                    np.sqrt((XYZSegmentDataArray[i,m,0] - XYZSegmentDataArray[i,m-1,0])**2 + \
                            (XYZSegmentDataArray[i,m,1] - XYZSegmentDataArray[i,m-1,1])**2 + \
                            (XYZSegmentDataArray[i,m,2] - XYZSegmentDataArray[i,m-1,2])**2)

        
        for m in range(0, NumPoints):
            LeftVector = np.zeros(3, dtype=np.float64)
            RightVector = np.zeros(3, dtype=np.float64)

            # Calculate vector direction
            if(m == 0):
                LeftVector[2] = ArbitraryEndSegLength
            else:
                LeftVector[0] = XYZSegmentDataArray[i,m,0] - XYZSegmentDataArray[i,m-1,0]
                LeftVector[1] = XYZSegmentDataArray[i,m,1] - XYZSegmentDataArray[i,m-1,1]
                LeftVector[2] = XYZSegmentDataArray[i,m,2] - XYZSegmentDataArray[i,m-1,2]

            if(m == NumPoints-1):
                RightVector[0] = XYZSegmentDataArray[i,m,0] - XYZSegmentDataArray[i,m-1,0]
                RightVector[1] = XYZSegmentDataArray[i,m,0] - XYZSegmentDataArray[i,m-1,0]
                RightVector[2] = ArbitraryEndSegLength
            else:                   
                RightVector[0] = XYZSegmentDataArray[i,m+1,0] - XYZSegmentDataArray[i,m,0]
                RightVector[1] = XYZSegmentDataArray[i,m+1,1] - XYZSegmentDataArray[i,m,1]
                RightVector[2] = XYZSegmentDataArray[i,m+1,2] - XYZSegmentDataArray[i,m,2]

            if(SegmentDistanceArray[m] > 0 and SegmentDistanceArray[m+1] > 0):
                VectorDifference = ((LeftVector[0]*RightVector[0]) + \
                                             (LeftVector[1]*RightVector[1]) + \
                                             (LeftVector[2]*RightVector[2]))/ \
                                             (SegmentDistanceArray[m]*SegmentDistanceArray[m+1])
             
                # arc cosine of 90 degrees is infinity
                if(int(VectorDifference) == 1):
                    AngleCurvatureArray[m] = 0
                else:
                    AngleBetweenSegs = np.arccos(VectorDifference)
                    # Curvature is based on angle and length of the two segments
                    AngleCurvatureArray[m] = (2.*AngleBetweenSegs)/ \
                          (SegmentDistanceArray[m] + SegmentDistanceArray[m+1])
  
                if(AngleCurvatureArray[m] == 0):
                    RadiusOfCurvatureArray[m] = 1.0e30
                else:
                    RadiusOfCurvatureArray[m] = 1/AngleCurvatureArray[m]

                # The X Difference is to see if a transition event has occurred in which the curve
                # in the x direction changes direction.  Only done in the x dimension, since it
                # accounts for the majority of the curvature.
                XDifferenceArray[m] = RightVector[0] - LeftVector[0]

                if(XDifferenceArray[m] < 0.0):
                    RadiusOfCurvatureArray[m] = -1.0 * RadiusOfCurvatureArray[m]

            else:
                AngleCurvatureArray[m] = 0.
                RadiusOfCurvatureArray[m] = 0.

            if(abs(AngleCurvatureArray[m]) > MaximumCurvature):
                MaximumCurvature = abs(AngleCurvatureArray[m])
                CurvatureZValue = XYZSegmentDataArray[i,m,2]

            if(abs(RadiusOfCurvatureArray[m]) < MinimumROC and RadiusOfCurvatureArray[m] != 0):
                MinimumROC = abs(RadiusOfCurvatureArray[m])
                ROCZValue = XYZSegmentDataArray[i,m,2]

        # Now we can calculate the power loss for this waveguide
        if(i == PrimarySegmentNumber):
            IsPrimary = 1
        else:
            IsPrimary = 0
        CalculatePowerLoss(ZSpatialPrecision, RadiusOfCurvatureArray, ArcRadiusDataArray[i], IsPrimary, \
                           PrimaryWaveguideLength, WPData)

        print "Minimum radius of curvature is " + str(MinimumROC) + \
            " at a Z distance of " + str(ROCZValue) + " microns."

    # Print Summary Values
    print "PROXIMITY AND HEIGHT VALUES"
    print "Closest Lateral Distance in the array is " + str(ClosestDistance) + " microns between segment " + \
          str(ClosestSegment1) + " and segment " + str(ClosestSegment2) + " at a Z distance of " + \
          str(ClosestZValue) + " microns."
    print "Maximum Y Value (closest to top) is " + str(MaximumYDistance) + " for segment " + \
          str(MaximumYDistSegment) + " at a Z distance of " + str(MaxYDistZValue) + " microns."
    print "Minimum Y Value (furthest from top) is " + str(MinimumYDistance)  + " for segment " + \
          str(MinimumYDistSegment) + " at a Z distance of " + str(MinYDistZValue) + " microns."

    print "************************************************************"

############################################################################################################
# CalculatePowerLoss
# This function attempts to estimate the power loss in each waveguide, based on calculations from
# Snyder and Love's textbook.  It uses the calculated waveguide radius of curvature at each point.

def CalculatePowerLoss(ZSpatialPrecision, ROCArray, ArcRadius, IsPrimary, PrimaryWaveguideLength, WPData):

    # The normalized power is set to 1.0 and then the loss from that input power is incremented
    # as z increases.  This gives the total power lost FROM CURVATURE ONLY at the end.

    NumSegments = len(ROCArray)
    PowerLossArray = np.zeros(NumSegments, dtype=np.float64)
    StartInputPower = 1.0
    TotalPowerLoss = 0.0
    PreviousROC = 0.0
    TotalGamma = 0.0
    TotalBendLossExp = 0.0
    TotalTransPower = 1.0
    TotalBendPower = 1.0
   
    # Loop through each segment piece that makes up the waveguide
    for i in range(0,NumSegments):
        CurrentROC = ROCArray[i]
        
        # Calculate the loss due to bends here (for a step profile)
        if(CurrentROC != 0 and abs(CurrentROC) != 1.0e30):
            # For Step profiles
##            CalculateSectionBendPowerLoss(CurrentROC, ZSpatialPrecision, WPData)

            # For Gaussian profiles
            CalculateSectionBendPowerLossGaussian(CurrentROC, ZSpatialPrecision, WPData)

            # To calculate bend loss for an HPO profile based on physical measurements
##            BendPower = HPOLookupBendLoss(CurrentROC, ZSpatialPrecision)
            
        else:
            GammaCoefficient = 0
            BendPower = 1.0
        
        TotalBendPower = TotalBendPower*BendPower

        # Calculate Transition Loss for splines
##        TransPowerLoss = CalculateSectionsTransitionLoss(CurrentROC, PreviousROC, WPData)
##        TotalTransPower = TotalTransPower * (1.0 - TransPowerLoss)
        
        PreviousROC = CurrentROC

    # Or for arc waveguide, just calculate the transition loss for the entire waveguide instead of above
    TotalTransPower = CalculateArcTransitionLoss(WPData, ArcRadius, IsPrimary)

    print "Power After Bend Loss: " + str(TotalBendPower)
    print "Power After Transition Loss: " + str(TotalTransPower)
    EndPower = TotalBendPower * TotalTransPower
    print "Total Power after Bend & Transition Losses: " + str(EndPower)
    # Last bit, incorporate the coupling and bend losses
    print "PrimaryWaveguideLength: " + str(PrimaryWaveguideLength)
    AddedLossPower = 0.91 *  EndPower * np.exp(-0.0075 * (PrimaryWaveguideLength/1000.))
    
    print "Total Power after all losses: " + str(AddedLossPower)


############################################################################################################
# HPOLookupBendLoss
# This function takes the values of bend loss per mm calculated from the right angle parameter scan result
# and return a bend loss value.  

def HPOLookupBendLoss(InputRadius, SegmentLength):
    # The normalized power is set to 1.0 and then the loss from that input power is incremented
    # as z increases.  This gives the total power lost FROM CURVATURE ONLY at the end.
    StartInputPower = 1.0
    CurrentROC = abs(InputRadius)

    # Radius and respective power loss per mm values
    RadiusLookupArray = np.array([10000.0,13300.0,16600.0,20000.0,23300.0,26600.0,30000.0,33300.0,36600.0,\
                                    40000.0], dtype=np.float64)
    BendLossLookupArray = np.array([0.437972203,0.18150836,0.078795534,0.040054237,0.017996858,0.013146017,\
                                    0.008745052,0.006998815,0.005485012,0.004484166], dtype=np.float64)
    
    # Calculate the loss due to bends here (for a step profile).  Interpolate between data points or
    # extrapolate from end points
    NumValues = len(BendLossLookupArray)-1
    FindLoss = interpolate.interp1d(RadiusLookupArray, BendLossLookupArray)
    BendLoss = 0.0

    # If the values are outside the range, they will just be linearly interpolated.  Most values will
    # be within this range.  Unlikely to have smaller radii, but lareger ones should not have much effect anyway
    if(CurrentROC < RadiusLookupArray[0]):
        Slope = (BendLossLookupArray[1]-BendLossLookupArray[0])/(RadiusLookupArray[0]-RadiusLookupArray[1])
        BendLoss = (RadiusLookupArray[0]-CurrentROC)*Slope + BendLossLookupArray[0]
    elif(CurrentROC > RadiusLookupArray[NumValues]):
        Slope = (BendLossLookupArray[NumValues]-BendLossLookupArray[NumValues-1])/\
                    (RadiusLookupArray[NumValues]-RadiusLookupArray[NumValues-1])
        BendLoss = BendLossLookupArray[NumValues] - (RadiusLookupArray[NumValues]-CurrentROC)*Slope
        if(BendLoss < 0):
            BendLoss = 0
    else:
        BendLoss = FindLoss(CurrentROC)

    # Scale this bend loss amount by relation of length to 1 mm (1000 microns)
    BendLoss = BendLoss * SegmentLength/1000.0

    # The power loss is Pout = Pin*(1-Loss)
    OutputPower = StartInputPower*(1.-BendLoss)
    return OutputPower
    
############################################################################################################
# CalculateSectionBendPowerLoss
# This function attempts to estimate the power loss in each waveguide, based on calculations from
# Snyder and Love's textbook.  It uses the calculated waveguide radius over a whole section with a
# constant radius and specified length.

def CalculateSectionBendPowerLoss(CurrentROC, SegmentLength, WPData):

    # The normalized power is set to 1.0 and then the loss from that input power is incremented
    # as z increases.  This gives the total power lost FROM CURVATURE ONLY at the end.
    StartInputPower = 1.0
    
    # Calculate the loss due to bends here (for a step profile)
    if(CurrentROC != 0):
        StepProfileCoefficient = (np.sqrt(np.pi)/(2*WPData.WaveguideRadius)) * \
                                 (np.sqrt(WPData.WaveguideRadius/CurrentROC))
        AreaCoefficient = WPData.UParameter**2/(WPData.NormalizedFrequency**2 * \
                            WPData.WParameter**1.5 * (scisp.kn(1,WPData.WParameter)**2))
        ExponentCoefficient = (-4./3)*(CurrentROC/WPData.WaveguideRadius) * \
                              ((WPData.WParameter**3 * WPData.DeltaParameter)/(WPData.NormalizedFrequency**2))
        GammaCoefficient =  StepProfileCoefficient * AreaCoefficient * np.exp(ExponentCoefficient)
    else:
        GammaCoefficient = 0

    # The power loss is Pout = Pin*exp(-Gamma*z)
    OutputPower = StartInputPower*np.exp(-1.*GammaCoefficient*SegmentLength)

    return OutputPower

############################################################################################################
# CalculateSectionBendPowerLossGaussian
# This function attempts to estimate the power loss in each waveguide, based on calculations from
# Snyder and Love's textbook.  It uses the calculated waveguide radius over a whole section with a
# constant radius and specified length.

def CalculateSectionBendPowerLossGaussian(CurrentROC, SegmentLength, WPData):

    # The normalized power is set to 1.0 and then the loss from that input power is incremented
    # as z increases.  This gives the total power lost FROM CURVATURE ONLY at the end.
    StartInputPower = 1.0
    
    # Calculate the loss due to bends here for a gaussian profile
    if(CurrentROC != 0):
        VMinus1 = WPData.NormalizedFrequency - 1.
        VPlus1 = WPData.NormalizedFrequency + 1.

        GaussCoefficient1 = (np.sqrt(np.pi)/(2.*WPData.WaveguideRadius)) * \
                                 (np.sqrt(WPData.WaveguideRadius/CurrentROC))
        GaussCoefficient2 = WPData.NormalizedFrequency**4/(VPlus1**2 * np.sqrt(VMinus1))
        ExponentCoefficient = (VMinus1**2 / VPlus1) - ((4./3)*(abs(CurrentROC)/WPData.WaveguideRadius) * \
                              ((VMinus1**3 * WPData.DeltaParameter)/(WPData.NormalizedFrequency**2)))
        GammaCoefficient =  GaussCoefficient1 * GaussCoefficient2 * np.exp(ExponentCoefficient)
    else:
        GammaCoefficient = 0

    # The power loss is Pout = Pin*exp(-Gamma*z)
    OutputPower = StartInputPower*np.exp(-1.*GammaCoefficient*SegmentLength)
    PowerLoss = 1.0 - OutputPower

    return OutputPower
  
############################################################################################################
# CalculateSectionsTransitionLoss
# This function attempts to estimate the power loss from transitions between two different radii of curvature
# from Snyder and Love's textbook.  This section calculation allows for two different ROC values to be set.
# Straight section calculated as 1.0e30

def CalculateSectionsTransitionLoss(Radius1, Radius2, WPData):

    TransitionCoefficient = 0.0
    if(Radius1 != 0.0 and Radius1 != 1.0e30 and Radius2 == 1.0e30):
        TransitionCoefficient = 1./Radius1**2
    elif(Radius2 != 0.0 and Radius2 != 1.0e30 and Radius1 == 0.0):
        TransitionCoefficient = 1./Radius2**2
    elif(Radius2 != 0.0 and Radius1 != 0.0 and Radius1 != 1.0e30 and Radius2 != 1.0e30):
        # The current ROC can be positive and the previous ROC can be negative and vice versa
        RadiiSum = 0.0
        if((Radius1 > 0.0 and Radius2 < 0.0) or (Radius2 > 0.0 and Radius1 < 0.0)):
            RadiiSum = abs(Radius1) + abs(Radius2)
        else:
            RadiiSum = Radius1 - Radius2
        TransitionCoefficient = (RadiiSum/(Radius1*Radius2))**2

    TransitionLoss = TransitionCoefficient * \
                   ((WPData.WaveguideRadius**2 * WPData.NormalizedFrequency**4)/ \
                    (8*WPData.DeltaParameter**2)) * \
                   (WPData.RadiusZero/WPData.WaveguideRadius)**6

    return TransitionLoss

############################################################################################################
# CalculateArcTransitionLoss
# This function attempts to estimate the power loss from transitions between two different radii of curvature
# from Snyder and Love's textbook.  Rather than evaluate a discrete curvature at the radius for each section,
# this routine takes into account that the radius is the same over the arc portions of the curve and will
# have three transition events for a primary waveguide and four for a secondary waveguide

def CalculateArcTransitionLoss(WPData, Radius, IsPrimary):

    TransitionCoefficient = 0.0
    TotalTransitionLoss = 1.0  # the remaining power after transition losses

    # All arc curves have two transitions from a straight section to the radius of the arcs at each end

    TransitionCoefficient = 1./Radius**2
    TransitionLoss = TransitionCoefficient * \
                   ((WPData.WaveguideRadius**2 * WPData.NormalizedFrequency**4)/ \
                    (8*WPData.DeltaParameter**2)) * \
                   (WPData.RadiusZero/WPData.WaveguideRadius)**6
    TotalTransitionLoss = TotalTransitionLoss * (1.0 - TransitionLoss)

    TransitionCoefficient = 1./Radius**2
    TransitionLoss = TransitionCoefficient * \
                   ((WPData.WaveguideRadius**2 * WPData.NormalizedFrequency**4)/ \
                    (8*WPData.DeltaParameter**2)) * \
                   (WPData.RadiusZero/WPData.WaveguideRadius)**6
    TotalTransitionLoss = TotalTransitionLoss * (1.0 - TransitionLoss)

    # Both primary and secondary waveguides have one transition where two opposite bends meet
    TransitionCoefficient = (2*Radius/(Radius*Radius))**2
    TransitionLoss = TransitionCoefficient * \
                   ((WPData.WaveguideRadius**2 * WPData.NormalizedFrequency**4)/ \
                    (8*WPData.DeltaParameter**2)) * \
                   (WPData.RadiusZero/WPData.WaveguideRadius)**6
    TotalTransitionLoss = TotalTransitionLoss * (1.0 - TransitionLoss)

    # Secondary waveguides have one additional opposite transtion
    if(IsPrimary != 1):
        TransitionCoefficient = (2*Radius/(Radius*Radius))**2
        TransitionLoss = TransitionCoefficient * \
                       ((WPData.WaveguideRadius**2 * WPData.NormalizedFrequency**4)/ \
                        (8*WPData.DeltaParameter**2)) * \
                       (WPData.RadiusZero/WPData.WaveguideRadius)**6
        TotalTransitionLoss = TotalTransitionLoss * (1.0 - TransitionLoss)
    
    return TotalTransitionLoss
               
############################################################################################################
# PLOT SEGMENTS

def PlotAllSegments(XYZSegmentDataArray):
    # Remember that z values are plotting on the y-axis here and y values on z-axis.  The plots obey the
    # right-hand rule, so this orients them in the proper direction.  y and z axes will be swapped on plot
    line0 = mlab.plot3d(XYZSegmentDataArray[0,:,0], XYZSegmentDataArray[0,:,2], XYZSegmentDataArray[0,:,1], \
                        tube_radius=10, tube_sides =12, colormap = 'Spectral', color = (0.5,0,0))
    line1 = mlab.plot3d(XYZSegmentDataArray[1,:,0], XYZSegmentDataArray[1,:,2], XYZSegmentDataArray[1,:,1], \
                        tube_radius=10, tube_sides =12, color = (0,0.5,0))
    line2 = mlab.plot3d(XYZSegmentDataArray[2,:,0], XYZSegmentDataArray[2,:,2], XYZSegmentDataArray[2,:,1], \
                        tube_radius=10, tube_sides =12, color = (1,0.5,0))
    line3 = mlab.plot3d(XYZSegmentDataArray[3,:,0], XYZSegmentDataArray[3,:,2], XYZSegmentDataArray[3,:,1], \
                        tube_radius=10, tube_sides =12, color = (0,0.5,0.9))
    line4 = mlab.plot3d(XYZSegmentDataArray[4,:,0], XYZSegmentDataArray[4,:,2], XYZSegmentDataArray[4,:,1], \
                        tube_radius=10, tube_sides =12, color = (0.8,0.4,0.8))
    line5 = mlab.plot3d(XYZSegmentDataArray[5,:,0], XYZSegmentDataArray[5,:,2], XYZSegmentDataArray[5,:,1], \
                        tube_radius=10, tube_sides =12, color = (1.0,0.75,0.0))
    line6 = mlab.plot3d(XYZSegmentDataArray[6,:,0], XYZSegmentDataArray[6,:,2], XYZSegmentDataArray[6,:,1], \
                        tube_radius=10, tube_sides =12, color = (0.125,0.6,0.65))
    line7 = mlab.plot3d(XYZSegmentDataArray[7,:,0], XYZSegmentDataArray[7,:,2], XYZSegmentDataArray[7,:,1], \
                        tube_radius=10, tube_sides =12, color = (1.0,1.0,0))
    mlab.show()
##    s = mayavi.script.engine.current_scene
##    s.scene.save('segmentcreate.jpg', size=(800,600))
    return

############################################################################################################
# CREATE FILE FEEDS
#
# This function creates the laser write file that will dictate to the laser write mechanism the positions
# to write each segment of the laser.  The segments will be ordered from bottom to top in the "y direction"
# however for this format, the z-direction becomes the y-axis for the laser, the y direction becomes the
# z-axis, and the x direction remains the same.

def CreateLaserWriteAndRSoftFiles(XYZSegmentDataArray):

    # Number of positions is the number of points in the z-direction
    now = datetime.datetime.now()
    NewDirName = "30mmchip" + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + \
                      str(now.minute) + str(now.second)
    print "Directory Created: " + NewDirName

    os.mkdir(NewDirName)
    os.chdir(NewDirName)

    # TURKEY
    FileLaserPoints = open('8GuideArcSideStepDesignBRedundant.txt','w')
    FileLaserPoints.write("8 segment, arc based sidestep, bends reversed, redundant output\n")
    SegmentOrderArray = np.array([3,1,5,0,6,2,4,7])

    # First loop goes through all segments
    NumberOfSegments = len(XYZSegmentDataArray[:,0,0])
    NumberOfPositions = len(XYZSegmentDataArray[0,:,2])
           
    for i in range(0, NumberOfSegments):
        # Use the segment order array for laser write file
        seg = SegmentOrderArray[i]
        print "Segment # " + str(seg)

        # Put a note here for the segment number
        FileLaserPoints.write("Segment " + str(i+1) + "\n")
        FileLaserPoints.write("\n")
        
        # A line needs to be added at the beginning to set the laser write at 2 mm from the input values

        # Change - all values multiplied by 0.001 to get mm values.
        XLead = XYZSegmentDataArray[seg,0,0]*-0.001
        ZLead = XYZSegmentDataArray[seg,0,1]*-0.001
        YLead = (-2.0)
        LineToWrite = "g1\tx\t" + str(XLead) + "\ty\t" + str(YLead) + "\tz\t" + str(ZLead) + "\n"
        FileLaserPoints.write(LineToWrite)
        
        for j in range(0, NumberOfPositions):
            XValue = XYZSegmentDataArray[seg,j,0]
            YValue = XYZSegmentDataArray[seg,j,2]
            ZValue = XYZSegmentDataArray[seg,j,1]

            # All x and z values should be multiplied by -1 for laser write file
            if (XValue != 0.0):
                XValue = XValue * -0.001
            if (ZValue != 0.0):
                ZValue = ZValue * -0.001
            YValue = YValue * 0.001
                
            XString = "%.4f" % XValue
            YString = "%.4f" % YValue
            ZString = "%.4f" % ZValue
            LineToWrite = "g1\tx\t" + XString + "\ty\t" + YString + "\tz\t" + ZString + "\n"
            FileLaserPoints.write(LineToWrite)

        # the end should have an empty line
        FileLaserPoints.write("\n")
            
    FileLaserPoints.close()
    print "FILE WRITE COMPLETE"

############################################################################################################ 
############################################################################################################ 
# MAIN PROGRAM VARIABLES
############################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, randn
import scipy.special as scisp
from enthought.mayavi import mlab
from enthought import mayavi
import datetime
import os
##from pytools import nmpfit

# Number of segments to be created
NumberOfSegments = 8

# Declare position and dimension data object
PosDimData = PositionDimensionDataObject(NumberOfSegments)
WavePropData = WaveguidePropertiesDataObject

# Optical Section Dimensions (microns)
PosDimData.BlockSectionWidth = 8000   # "X" dimension   #Physical chip width 8000
PosDimData.BlockSectionHeight = 1000  # "Y" dimension   #Physical chip height 1100
PosDimData.BlockSectionLength = 30000 # "Z" dimension

# Number of segments to be created
PosDimData.NumberOfSegments = NumberOfSegments

# Initialize and calculate waveguide properties data
CalculateWaveguideProperties(WavePropData)

############################################################################################################
# START POINTS
############################################################################################################

# New Design, 8 hole, blue dots of diagram
PosDimData.PathXStartPosArray = [-77.94,-51.96,-51.96,0.0,25.98,51.96,77.94,77.94] 
PosDimData.PathYStartPosArray = [15.0,60.0,-60.0,90.0,-45.0,60.0,15.0,-45.0]

############################################################################################################
# END POINTS
############################################################################################################

# 8 segment redundant array, shifted in +x direction by 5 mm
##PosDimData.PathXEndPosArray = [4875.0,4375.0,5375.0,4125.0,5625.0,4625.0,5125.0,5875.0]   # Splines
##PosDimData.PathXEndPosArray = [4125.0,4375.0,5375.0,4625.0,5625.0,4875.0,5125.0,5875.0]   # Arcs
PosDimData.PathXEndPosArray = [4125.0,4375.0,4625.0,4875.0,5125.0,5375.0,5625.0,5875.0]     # Reverse Arcs
PosDimData.PathYEndPosArray = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


############################################################################################################

# Z Direction finite measurement amount (micron)
# The layout of the program is that the chip is modelled as slices of x-y coordinates for each segment
# created in the Z direction.  This variable specifies the precision (grain) to calculate with.  This
# will increase the number of slices created, increasing the precision, but also the computing time.

# For now - make this an even multiple of the block section length
PosDimData.ZSpatialPrecision = 25.0
PosDimData.NumberZSlices = (PosDimData.BlockSectionLength/PosDimData.ZSpatialPrecision)

# length precision is how close each segment lengths must be (microns)
PosDimData.SegmentLengthPrecision = 0.1

# Minimum Path Separation Distance
# This variable is an optional input parameter.  The program will do an optional check to see that
# the created paths are separated from each other by this distance.  The program is optimised to
# maximize the separation distance, so if it is not specified (set to zero), it will assume it is fine.
PosDimData.MinimumPathSeparationDist = 30.0

# The calculated path distance is the length that each segment needs to have.  Usually determined
# by the primary segment.  If created as an array, the value can be returned from functions.

# The current Dragonfly has a 1 mm straight section at either end of the chip to maximize the light inpuPosDimData.PathXEndPosArray = [4125.0,4375.0,5375.0,4625.0,5625.0,4875.0,5125.0,5875.0]   # Arcs
# into the chip.  These variables will specify how large of a straight section on either end, as well
# as a straight section in the middle to bridge the creation of a double spline section
# (in microns)
PosDimData.EdgeStraightSectionLength = 1000.0
PosDimData.CenterBridgeLength = 6000.0

# This is the 4 dimensional array that will hold the complete set of XYZ Data for all splines
# [X,Y,Z,Segment] - X,Y,Z = 3D coordinates, Segment = number of the segment
XYZSegmentDataArray = np.zeros((PosDimData.NumberOfSegments, PosDimData.NumberZSlices+1, 3), dtype=np.float64)
ArcRadiusDataArray = np.zeros(PosDimData.NumberOfSegments, dtype=np.float64)

############################################################################################################
# PROGRAM DESIGN
############################################################################################################

# To calculate the path, the program uses the spline function.  This takes the starting and end
# points of the path, then returns the centerpoints of the function for each z-slice.

############################################################################################################
# Main logic loops
############################################################################################################

# This is the main program that calls the sub functions and keeps track of the flow.
# Input/Output Coordinates
CalculateInputOutputDistance(PosDimData)

print PosDimData.XDistanceArray
print PosDimData.YDistanceArray
print PosDimData.DirectDistanceArray

PosDimData.PrimarySegmentNumber = DeterminePrimarySegments(PosDimData.DirectDistanceArray)

# ARC BASED CODE
CreatePrimaryArcSegment(PosDimData, XYZSegmentDataArray, ArcRadiusDataArray)
DetermineSecondarySegments(PosDimData)
CreateSecondaryArcSegments(PosDimData, XYZSegmentDataArray, ArcRadiusDataArray)

# SPLINE BASED CODE
##primaryOK = CreatePrimarySegments(PosDimData, XYZSegmentDataArray)
##DetermineSecondarySegments(PosDimData)
##secondaryOK = CreateSecondarySegments(PosDimData, XYZSegmentDataArray)

SegmentDistanceCurvatureCheck(XYZSegmentDataArray, PosDimData, WavePropData, ArcRadiusDataArray)

#Show the final 3d plot
PlotAllSegments(XYZSegmentDataArray)

# Create files if specified
##CreateLaserWriteAndRSoftFiles(XYZSegmentDataArray)


print "*** END ***"
