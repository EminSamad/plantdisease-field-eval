# Laboratory-to-field evaluation of plant disease classifiers

Code and class-mapping tables for the paper **"Does Laboratory Accuracy Predict Field Performance? Cross-Dataset Evaluation of ResNet50, EfficientNet-B0 and Swin-T for Plant Disease Classification"** (Emin Samadzade, 2026, manuscript in preparation).

ImageNet-pretrained ResNet50, EfficientNet-B0 and Swin-T are fine-tuned on PlantVillage (laboratory images, 38 classes) and then tested, without any adaptation, on two datasets of images collected outside the laboratory: PlantDoc (27 shared classes) and PlantWild v1 (33 shared classes). Each architecture is trained several times to measure run-to-run variability.

## Main results reported in the paper

Mean ± sample SD over three training runs per architecture (accuracy, %). "Restricted" means the model chooses only among the classes shared with the target dataset.

| Model | PlantVillage test | PlantDoc restricted | PlantWild restricted |
|---|---|---|---|
| EfficientNet-B0 | 99.68 ± 0.02 | 20.54 ± 0.84 | 14.03 ± 0.25 |
| ResNet50 | 99.51 ± 0.04 | 22.17 ± 0.43 | 15.57 ± 0.25 |
| Swin-T | 99.40 ± 0.14 | 26.37 ± 1.80 | 18.42 ± 1.18 |

The architecture ranking on PlantVillage is reversed on both field datasets. With three runs per model, and with architecture, size and pretraining recipe varying together, the paper does not conclude that one architecture family generalises better than another. See the paper for limitations.

## Repository layout

```
src/common.py           transforms, class mappings, data split, models, prediction helper
src/train.py            two-stage fine-tuning on PlantVillage (resumable)
src/evaluate_field.py   tests trained models on PlantDoc / PlantWild
src/analyze.py          mean ± SD tables, Welch tests, species-vs-disease error split
src/gradcam.py          optional Grad-CAM illustration
mappings/               PlantVillage class list and the two class-correspondence tables
tests/test_mappings.py  consistency check of the mapping files (pure Python)
```

## Datasets (not redistributed here)

Download them from their original sources and check each licence before use.

| Dataset | Source | Licence stated by the source |
|---|---|---|
| PlantVillage ("color" subset) | https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset | CC BY-NC-SA 4.0 |
| PlantDoc (classification folders) | https://github.com/pratikkayal/PlantDoc-Dataset | CC BY 4.0 |
| PlantWild v1 (`plantwild.zip`) | https://huggingface.co/datasets/uqtwei2/PlantWild | CC BY-NC-ND 4.0 |

Please cite the original dataset papers (Hughes and Salathé 2015; Mohanty et al. 2016; Singh et al. 2020; Wei et al. 2024).

## Class mapping

`mappings/plantdoc_mapping.csv` and `mappings/plantwild_mapping.csv` map each target class to a PlantVillage class. The mapping involves judgement, in particular:

- *bell pepper leaf spot* → `Pepper,_bell___Bacterial_spot` (both datasets);
- PlantWild *citrus greening disease* → `Orange___Haunglongbing_(Citrus_greening)`;
- PlantWild *grape leaf spot* is excluded because its correspondence to a PlantVillage class could not be established.

Target-dataset labels are used as provided and were not verified by plant pathologists. Run `python tests/test_mappings.py` to check the files.

## Usage

```bash
pip install -r requirements.txt

# 1. PlantVillage (e.g. via the Kaggle CLI), PlantDoc, PlantWild
kaggle datasets download -d abdallahalidev/plantvillage-dataset && unzip -q plantvillage-dataset.zip -d data
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
# PlantWild: download plantwild.zip from the Hugging Face page above and unzip it to ./plantwild
# (the folder that contains images/ is then plantwild/plantwild)

# 2. Train (three runs per architecture; GPU strongly recommended)
for arch in resnet50 efficientnet_b0 swin_t; do
  for seed in 0 1 2; do
    python src/train.py --arch $arch --seed $seed \
      --data-dir "data/plantvillage dataset/color" --out-dir runs
  done
done

# 3. Evaluate on the field datasets
python src/evaluate_field.py --out-dir runs \
  --plantdoc-dir PlantDoc-Dataset --plantwild-dir plantwild/plantwild

# 4. Summaries and tests
python src/analyze.py --out-dir runs
```

Protocol (as in the paper): images resized to 224 × 224 with ImageNet normalisation; random flip, rotation up to 15° and colour jitter 0.2 for training; 70/15/15 image-level split with fixed seed 42, identical for every run; 3 epochs training only the new head (Adam, lr 1e-3) followed by 5 epochs of full fine-tuning (Adam, lr 1e-4); batch size 32; checkpoint chosen by validation macro-F1; mixed precision on GPU.

## Reproducibility notes

- **The numbers in the paper were produced with notebook code (Google Colab), which these scripts consolidate.** The scripts were syntax-checked, the mapping files tested, and `analyze.py` smoke-tested on synthetic predictions. They have **not yet been re-run end to end on a GPU with the real datasets**, so the paper's numbers have not been reproduced with this exact code.
- In the paper each architecture has an *initial run* plus runs with explicit seeds 1 and 2. The seed of the initial runs was not recorded separately for each architecture. The initial ResNet50 run used float32, the other runs mixed precision. Running seeds 0, 1 and 2 with these scripts will therefore give numbers close to, but not identical with, the paper.
- GPU operations are not fully deterministic, so repeated runs can differ slightly.
- The notebooks used different prediction file names; `evaluate_field.py` writes `preds_<dataset>_<tag>.npz` with arrays `y`, `p_all` (38-way) and `p_restricted`.
- Library versions recorded in the Google Colab environment: PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, scikit-learn 1.6.1, SciPy 1.16.3. The version of pytorch-grad-cam was not recorded.

## Citation

Citation information will be added after publication.

## Licence

Code: MIT (see `LICENSE`). Datasets keep their own licences.
