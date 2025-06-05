import cv2
import numpy as np

'''
ASSESS IMAGE FIELD QUALITY PARAMETER BASED ON YOLO MODEL
This file presents a series on functions responsible
to assess if the image matches all description for
image field quality. This parameter is graded according
to this table:
https://doi.org/10.1371/journal.pdig.0000454.t002
'''

# Determine if the OD/Fovea is on the right or left side of the image
# Parameters:
#   - imageInfo: object with info about the image passed on the model
#   - imageWidth: width of the full image
def determineStructureDirection(imageInfo, imageWidth):
    xywh = imageInfo.boxes.xywh

    return 'RIGHT' if xywh[0][0].item() > (imageWidth / 2) else 'LEFT'

# Get the right/left most coordinate of the OD
# Parameters:
#   - odSide: side of the retina that the OD is located at
#   - imageInfo: object with info about the image passed on the model
def getODCornerCoords(odSide, imageInfo):
    xywh = imageInfo.boxes.xywh
    odY = xywh[0][1].item()

    if odSide == 'RIGHT':
        odX = xywh[0][0].item() + (xywh[0][2].item() / 2)
    elif odSide == 'LEFT':
        odX = xywh[0][0].item() - (xywh[0][2].item() / 2)

    return odX, odY

# Get fovea center coordinate
# Parameters:
#   - imageInfo: object with info about the image passed on the model
def getFoveaCoords(imageInfo):
    xywh = imageInfo.boxes.xywh

    odX = xywh[0][0].item()
    odY = xywh[0][1].item()

    return odX, odY

# Determine distance from the structure to the closest edge
# Optic Disc: distance from its border to nasal edge
# Fovea: distance from its center to temporal edge
# Parameters:
#   - imagePath: path to the image to analyze
#   - imageInfo: object with info about the image passed on the model
#   - structure: 'od' or 'fovea'
def getEdgeDistance(imagePath, imageInfo, structure='od', write_tresh=False):
    image = cv2.imread(imagePath)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageWidth = image.shape[1]
    
    direction = determineStructureDirection(imageInfo, imageWidth)

    if structure == 'od':
        odX, odY = getODCornerCoords(direction, imageInfo)
    elif structure == 'fovea':
        odX, odY = getFoveaCoords(imageInfo)

    # threshold to isolate the retina
    _, thresh = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)

    if write_tresh:
        cv2.imwrite('./threshVisualization.jpg', thresh)

    # run horizontaly on the image to find the edge
    distance = 0
    if direction == 'LEFT':
        for x in range(0, int(odX)):
            if thresh[int(odY), x] != 0:
                distance = odX - x
                break
    else:
        for x in range(imageWidth-1, 0, -1):
            if thresh[int(odY), x] != 0:
                distance = x - odX
                break

    print(f"A distancia de {structure} até o edge é {distance:.2f}")
    
    return distance