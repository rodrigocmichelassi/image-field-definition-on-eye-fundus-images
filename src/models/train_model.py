import os
import argparse
from ultralytics import YOLO

'''
    Training and evaluating YOLO model for 
    optic disc and fovea detection. On 
    command-line input specify opticdisc1, 
    opticdisc2 or fovea to select which model
    to train
'''

# Separate args to train according to dataset wished to use
def getArgs(args):
    dataset_path = '/home/rodrigocm/datasets'
    baseline_path = '/home/rodrigocm/research/YOLO-on-fundus-images/data'

    if args.dataset == 'opticdisc1':
        dataset_path = os.path.join(dataset_path, 'opticDiscDetection/baseline1')
        baseline_path = os.path.join(baseline_path, 'od_baseline1.yaml')
        name = 'od_baseline1'

    elif args.dataset == 'opticdisc2':
        dataset_path = os.path.join(dataset_path, 'opticDiscDetection/baseline2')
        baseline_path = os.path.join(baseline_path, 'od_baseline2.yaml')
        name = 'od_baseline2'

    elif args.dataset == 'fovea':
        dataset_path = os.path.join(dataset_path, 'foveaDetection/images')
        baseline_path = os.path.join(baseline_path, 'fovea_baseline.yaml')
        name = 'fovea_baseline'
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")

    print(f"Traning for: {name}")

    return dataset_path, baseline_path, name

# Run the model on validation and test files, and save results
def test(model, baseline, baseline_name):
    print("-*-*-*- Resultados de Validação -*-*-*-")
    model.val(
        data=baseline,
        max_det=1
    )

    if baseline_name != 'fovea_baseline':
        print("-*-*-*- Resultados de Teste -*-*-*-")
        model.val(
            data=baseline,
            split='test',
            max_det=1
        )

# Train model and run results for test images
def main(args):
    model = YOLO("yolo11n.pt")
    dataset_path, baseline_path, name = getArgs(args)

    model.train(
        data=baseline_path, 
        epochs=40,
        patience=10,
        optimizer='Adam',
        device=args.gpu,
        save=True,
        name=name
    )
 
    model.predict(
        source=f'{dataset_path}/test/images',
        conf=0.5,
        max_det=1,
        save=True
    )

    test(model, baseline_path, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a object detection model")
    parser.add_argument('--dataset', type=str, default='opticdisc1', help='YOLO dataset to be used for training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to run the model training')

    args = parser.parse_args()

    main(args)
