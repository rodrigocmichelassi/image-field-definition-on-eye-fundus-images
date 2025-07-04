import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon

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

# Return BBox needed information from a structure
# Parameters:
#   - imageInfo: bbox information
#   - structure: 'od' or 'fovea'
def getBboxInformation(imageInfo, structure):
    xywh = imageInfo.boxes.xywh

    cx = xywh[0][0].item()
    cy = xywh[0][1].item()

    if structure == 'fovea':
        return np.array([cx, cy])
    
    width = xywh[0][2].item()
    height = xywh[0][3].item()

    return np.array([cx, cy]), width, height

# Extract nasal point from OD, given a
# Fovea-OD unit vector, pointing to the
# nasal edge
# Parameters:
#   - odCenter: np array with optic disc center coordinates
#   - width: optic disc width given by bbox
#   - height: optic disc height given by bbox
#   - nasalUnitVector: Fovea-OD unit vector
def getNasalPointOnOD(odCenter, width, height, nasalUnitVector):
    cx, cy = odCenter
    ux, uy = nasalUnitVector

    # retorna 1000 pontos, no intervalo [0, 2pi] (intervalo de angulos da elipse)
    ts = np.linspace(0, 2*np.pi, 1000)
    xs = (width/2) * np.cos(ts)
    ys = (height/2) * np.sin(ts)

    # projecoes na direção nasal (angulo OD e Fovea)
    projections = xs * ux + ys * uy

    # o maior valor aponta a projeção mais próxima do vetor nasal
    idx = np.argmax(projections)

    x_nasal = cx + xs[idx]
    y_nasal = cy + ys[idx]

    return x_nasal, y_nasal, ts[idx]

# Calculate distance from a point to its
# closest edge (temporal or nasal)
# Parameters:
#   - image: cv2 image object
#   - unitVector: nasal or temporal OD-Fovea unit vector
#   - retinalPoint: point from retina to calculate distance from
def calculateDistance(image, unitVector, retinalPoint):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)   # contorno mais externo da retina

    retina_edge = Polygon(main_contour.squeeze())

    if not retina_edge.contains(Point(retinalPoint)):
        print("Ponto nasal já está na borda da retina ou fora dela — distância = 0")
        return 0.0

    line = LineString([retinalPoint, retinalPoint + 3000 * unitVector])
    intersection = line.intersection(retina_edge.boundary)

    if intersection.geom_type == 'MultiPoint':
        nasalEdgePoint = min(intersection.geoms, key=lambda pt: pt.distance(Point(retinalPoint)))
    elif intersection.geom_type == 'Point':
        nasalEdgePoint = intersection
    else:
        raise ValueError("Interseção inesperada")

    distance = Point(retinalPoint).distance(nasalEdgePoint)
    
    return distance

# Calculate distance from temporal edge
# to the Fovea and from nasal edge to 
# the Optic Disc.
# Parameters:
#   - imagePath: path to the image file
#   - odInfo: BBox information for OD
#   - foveaInfo: BBox information for Fovea
#   - saveNasalPoint: wheter to save or not nasal point image   
#   - outputPath: where to save nasal point image
def getEdgeDistanceReal(imagePath, odInfo, foveaInfo, saveNasalPoint=True, outputPath="./src/data/images"):
    image = cv2.imread(imagePath)

    odCenter, width, height = getBboxInformation(odInfo, structure='od')
    foveaCenter = getBboxInformation(foveaInfo, structure='fovea')
    
    odFoveaVector = foveaCenter - odCenter
    temporalUnitVector = odFoveaVector / np.linalg.norm(odFoveaVector)
    nasalUnitVector = -temporalUnitVector

    nasalX, nasalY, theta = getNasalPointOnOD(odCenter, width, height, nasalUnitVector)
    nasalPoint = np.array([nasalX, nasalY])
    
    if saveNasalPoint is True:
        _, fileName = os.path.split(imagePath)

        name, ext = os.path.splitext(fileName)
        output_filename = f'{name}_nasal{ext}'

        cv2.circle(image, (int(nasalX), int(nasalY)), radius=5, color=(0,0,255), thickness=-1)
        cv2.imwrite(os.path.join(outputPath, output_filename), image)

    nasalDistance = calculateDistance(image, nasalUnitVector, nasalPoint)
    temporalDistance = calculateDistance(image, temporalUnitVector, np.array(foveaCenter))

    return nasalDistance, temporalDistance, theta