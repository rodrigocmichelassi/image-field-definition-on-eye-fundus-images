import os
import cv2
import argparse
from ultralytics import YOLO
from src.utils.assess_quality import getEdgeDistance

'''
RUN YOLO MODEL ON RETINAL IMAGE
This file is responsible for running a trained
yolo model on a retinal image, in order to 
detect the optic disc location on the image
and extract information, with special
attention to the width of the bounding box
(Disc Diameter DD)s
'''

# Print obtained bounding box coordinates in different formats
# Parameters:
#   - results: object with info about the image passed on the model
def showResults(results):
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        print(xywh)
        xywhn = result.boxes.xywhn  # normalized
        print(xywhn)
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        print(xyxy)
        xyxyn = result.boxes.xyxyn  # normalized
        print(xyxyn)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        print(names)
        confs = result.boxes.conf  # confidence score of each box
        print(confs)

# Given a path to trained weights, load a yolo model
# Parameters:
#   - weightsPath: path to weights .pt to load model
def loadModel(weightsPath):    
    model = YOLO(os.path.join(weightsPath, 'best.pt'))
    return model

# Run the model on an image and return results
# Parameters:
#   - imagePath: path to an image to get structure coordinates
#   - model: model instance
def getCoordinates(imagePath, model):
    _, fileName = os.path.split(imagePath)
    imageName, _ = os.path.split(fileName)

    destPath = './src/data/images'

    model.predict(imagePath, save_txt=False, project=destPath, name=imageName)

    results = model(imagePath)

    return results

# Draw the bounding boxes on both od and fovea
# Parameters:
#   - image: the openCV image to draw the boxes
#   - results: object list, represent the output from the model
#   - color: bbox color
#   - objectName: bbox name
def draw_boxes(image, results, color, objectName=""):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"{objectName}{conf:.2f}"

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_offset_x = x1
        text_offset_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5

        # Draw the rectangle for the text
        cv2.rectangle(image,
                      (text_offset_x, text_offset_y - text_height - baseline),
                      (text_offset_x + text_width, text_offset_y + baseline),
                      color,
                      -1)  # Filled

        # Write text
        cv2.putText(image,
                    label,
                    (text_offset_x, text_offset_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA)

# Get prediction results from running od and fovea detection
# Parameters:
#   - imagePath: path to the image to run the models
#   - odModel: model instance to detect OD
#   - foveaModel: model instance to detect fovea
#   - saveImg: bool, save or not the image
#   - output_path: where to save the image
def getPredictions(imagePath, odModel, foveaModel, saveImg, output_path="./src/data/images",):
    _, fileName = os.path.split(imagePath)
    image = cv2.imread(imagePath)

    odResults = odModel(imagePath)
    foveaResults = foveaModel(imagePath)

    draw_boxes(image, odResults, color=(0, 170, 0), objectName="Optic Disc: ")
    draw_boxes(image, foveaResults, color=(255, 0, 0), objectName="Fovea: ")

    if saveImg is True:
        cv2.imwrite(os.path.join(output_path, fileName), image)

    return odResults, foveaResults

# Given a dataset path, run the model on all the images of the dataset
# Parameters:
#   - dataPath: path to the dataset
#   - model: model instance
def runModelOnDataset(args, saveImg=True):
    imagesInfo = []

    odModel = loadModel(args.od_weights)
    foveaModel = loadModel(args.fovea_weights)

    for image in os.listdir(args.data_path):
        _, extension = os.path.splitext(image)
        
        if extension == '.jpg':
            results = getCoordinates(os.path.join(args.data_path, image), odModel)
            imagesInfo.append(results)

    showResults(imagesInfo[0])
    
    return imagesInfo

# Given an image path, run the model on the specific image
# Parameters:
#   - args: input arguments
#   - saveImg: bool, save or not the image
#   - showResults: bool, enable debug on image run
def runModelOnImage(args, saveImg=True, showResults=False):
    imagePath = os.path.join(args.data_path, f'{args.image}.jpg')
    
    odModel = loadModel(args.od_weights)
    foveaModel = loadModel(args.fovea_weights)

    # Run the model on an image to locate optic disc and fovea
    odInfo, foveaInfo = getPredictions(imagePath, odModel, foveaModel, saveImg)
    
    if showResults is True:
        showResults(odInfo)
        showResults(foveaInfo)
    
    discDiameter = odInfo[0].boxes.xywh[0][2].item()
    print(f'O diâmetro do disco óptico é: {discDiameter:.2f}')

    # Detect the distance between the edges and the optic disc/fovea
    nasalDistance = getEdgeDistance(imagePath, odInfo[0], structure='od')
    temporalDistance = getEdgeDistance(imagePath, foveaInfo[0], structure='fovea')

    if nasalDistance < discDiameter:
        print(f"The image is inadequate. Criteria: [1] the distance from the OD to the nasal edge ({nasalDistance:.2f}) is lower than 1DD ({discDiameter:.2f}).")

    if temporalDistance < 2*discDiameter:
        print(f"The image is inadequate. Criteria: [2] the distance from the Macular Center to the temporal edge ({temporalDistance:.2f}) is lower than 2DD ({2*discDiameter:.2f}).")

# Main execution, decides how to run the model
def main(args):
    if args.image is not None:
        runModelOnImage(args, saveImg=True)

    elif args.image is None:
        runModelOnDataset(args, saveImg=False)

# python main.py --image img01233   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assess image field definition (BRSet)")

    # img = img10829
    parser.add_argument('--image', type=str, default=None, help='Fundus image name')
    parser.add_argument('--data-path', type=str, default='/scratch/diogo.alves/datasets/brset/physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos/', help='Fundus image dataset path')
    parser.add_argument('--od-weights', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/src/models/runs/detect/od_baseline1/train_results/weights', help='OD detection weights path')
    parser.add_argument('--fovea-weights', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/src/models/runs/detect/fovea/train_results/weights', help='Fovea detection weights path')

    args = parser.parse_args()

    main(args)