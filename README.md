# Solving Image Field criteria for BRSet dataset

In case of any questions concerning this project, reach out to me in rodrigo (dot) michelassi (at) usp (dot) br.

## Code Explanation

This presented code follows the research paper [link when available], in which we partially solve the Image Field quality assessment protocol for the BRSet dataset. We use two YOLO models for Fovea and Optic Disc detection, and calculate the distances between these structures to the retinal edges, based on a geometric approach. As for this moment, the data preparation code is not available on this repository, since it is not well documented, but may be added in the future. For reproducibility, every step of this research is explained on [my personal blog](https://rodrigocmichelassi.github.io/categories/retinal-fundus-image-quality-assessment/), step by step.

### Training a model

Given the prepared data for YOLO training, you can train models for optic disc or fovea detection by running:

```bash
cd src/models
python train_model.py \
    --dataset <opticdisc1, opticdisc2 or fovea> \
    --gpu <gpu_index>
```

### Running BRSet inferences

With the trained model, it is now possible to run inferences for BRSet. Inferences can be generated for a single image or for the entire dataset.

```bash
python main.py \
    --image <img_id> \
    --data-path <path_to_brset_data> \
    --od-weights <path_to_od_detection_model_weights> \
    --fovea-weights <path_to_fovea_detection_model_weights> \
```

If wishing to run inferences for the entire dataset, leave the `--image` field empty. In this mode, all inferences will be saved in `/data/retinalInformation.csv`.

### Generating statistics

After generating inferences for BRSet, one may wish to visualize the data distribution. For this, it is possible to generate histograms, regarding the distribution of the Optic Disc - Nasal Edge and Fovea - Temporal edge distances, or the angle $$\theta$$ between Optic Disc and Fovea. To do so, run:

```bash
cd src/utils
python dataset_analysis.py \
    --dataset-size <size of the dataset>
    --data <path_to_csv_file>
    --save-path <path_to_save_histograms>
```
