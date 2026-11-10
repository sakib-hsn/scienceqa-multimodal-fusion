# ScienceQA Multimodal Fusion

This project tests whether adding images improves multiple-choice science question answering on [ScienceQA-IMG](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG). It compares a text-only baseline with two ways of combining text and images:

- **Text only:** Uses BERT features from the question, optional hint and each answer choice.
- **Early fusion:** Combines the BERT features with a single summary of the image from ViT.
- **Attention fusion:** Lets each answer choice attend to individual ViT image patches.

BERT and ViT are frozen. Only the smaller models that score the answer choices are trained.

## Reproduce the results in Google Colab

1. Open [`scienceqa_multimodal_fusion.ipynb`](scienceqa_multimodal_fusion.ipynb) on GitHub and click **Open in Colab**. If the badge is missing, download the notebook and upload it to [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime → Change runtime type** and select a **T4 GPU**. The reported results were produced on a Tesla T4.
3. Select **Runtime → Run all**. The first cell installs `datasets`. The notebook also uses PyTorch, Transformers, Pillow, NumPy, pandas, Matplotlib and tqdm.
4. On the first run, Colab downloads ScienceQA-IMG and the pretrained ViT and BERT weights. No manual dataset download is required.
5. After the last cell finishes, download `final_results.csv` and `test_accuracy.png` from Colab’s **Files** panel if you want to keep them.

Run the cells in order in the same session. The notebook keeps extracted features and model checkpoints in memory, so if the runtime restarts, run it again from the beginning.

## Data and training

The notebook checks for unreadable images and invalid questions or answers. It removes validation and test examples that repeat an exact image or the same normalised text input from an earlier retained split. The resulting splits contain **6,218 training**, **1,105 validation** and **1,050 test** questions. Answer explanations are excluded from the inputs.

Each model is trained with seeds **42, 43 and 44**, using AdamW with a learning rate of **0.001** and a batch size of **32**. Training runs for up to **15 epochs**, with early stopping after **four epochs** without an improvement in validation loss. The checkpoint with the lowest validation loss for each run is used for the final test evaluation.

Small differences in scores and timing are possible across Colab runtimes. The dataset and pretrained models are loaded by name, so changes to their hosted versions could also affect reproducibility.

## Results

| Model | Mean test accuracy | Seed SD (percentage points) | Mean test loss | Cached inference (ms/question) |
|---|---:|---:|---:|---:|
| Text only | 63.17% | 0.24 | 0.686 | 0.0188 |
| Early fusion | 63.94% | 0.61 | 0.681 | 0.0262 |
| Attention fusion | **74.41%** | 3.25 | **0.569** | 0.1236 |

Attention fusion improved accuracy by **11.24 percentage points** over text only (95% image-group bootstrap interval: **9.17 to 13.20 points**). Early fusion improved by **0.76 points**, but its interval included zero. Attention fusion took about **4.7 times as long** as early fusion for cached inference. These timings are per question within a batch and exclude feature extraction.

The notebook also includes learning curves, results by subject and grade, an image-shuffling check, and an example of a confident wrong answer.

## Credits

- Dataset: [ScienceQA-IMG](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG), based on the [original ScienceQA project](https://scienceqa.github.io/).
- Pretrained models: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) and [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased).
