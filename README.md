<h1 align="center">Audio-Visual Source Separation(AVSS)</h1>

<p align="center">
  <a href="#about">About</a> •
    <a href="#example">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
   <a href="#final-results">Final results</a> •
  <a href="#credits">Credits</a> •
   <a href="#authors">Authors</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the end-to-end pipeline for solving AVSS task with PyTorch. The model was implemented
is [RTFS-Net](https://arxiv.org/abs/2309.17189).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/project_avss).

See a [report](https://github.com/free001style/AVSS/report.pdf) for more information.

## Examples

<details>

<summary>Examples</summary>

Mixed audio |
:-: |
<video src='https://github.com/user-attachments/assets/1eef968b-f39d-4b5e-9560-bc453068b891' width=180/> |

- AVSS model:

    speaker 1 | speaker 2
    :-: | :-:
    <video src='https://github.com/user-attachments/assets/e78f197e-958f-497b-80a0-a6be51363074' width=180/> | <video src='https://github.com/user-attachments/assets/341adeee-ecb6-4600-b66e-a5c531ebebd7' width=180/>

- Audio-only model:

    speaker 1 | speaker 2
    :-: | :-:
    <video src='https://github.com/user-attachments/assets/afdc7d50-1408-4779-b3e2-a3cc5de54c25' width=180/> | <video src='https://github.com/user-attachments/assets/4d35f835-24a7-4919-a4eb-41b082ac2dea' width=180/>


</details>

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n AVSS python=3.11

   # activate env
   conda activate AVSS
   ```

1. Install all required packages.

   ```bash
   pip install -r requirements.txt
   ```
2. Download model checkpoint, vocab and language model.

   ```bash
   python download_weights.py
   ```

## How To Use

### Inference

<details>
<summary> 1. To separate two-speaker mixed audio, audio directory should have the following format:</summary>

```
NameOfTheDirectoryWithTestDataset
├── audio
    ├── mix
        ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
        ├── FirstSpeakerID2_SecondSpeakerID2.wav
        .
        .
        .
        └── FirstSpeakerIDn_SecondSpeakerIDn.wav
```

Run the following command:

```bash
python inference.py \
    datasets=inference_custom \
    datasets.test.data_dir=TEST_DATASET_PATH \
    inferencer.save_path=SAVE_PATH \
    model=no_video_rtfs \
    inferencer.from_pretrained='saved/other/no_video_model.pth'
```

where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.

</details>

<details>
<summary>2. To separate two-speaker mixed audio using reference mouth recordings, audio and video directories should have the
   following format:</summary>

```
NameOfTheDirectoryWithTestDataset
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```

Run the following command:

```bash
python inference.py \
    datasets=inference_custom \
    inferencer.save_path=SAVE_PATH \
    datasets.test.data_dir=TEST_DATASET_PATH
```

where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.

</details>

<details>
<summary> 3. To separate two-speaker mixed audio and evaluate results against ground truth separation, audio directory should
   have the following format:</summary>

```
NameOfTheDirectoryWithTestDataset
├── audio
    ├── mix
    │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
    │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
    │   .
    │   .
    │   .
    │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
    ├── s1 # ground truth for the speaker s1, may not be given
    │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
    │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
    │   .
    │   .
    │   .
    │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
    └── s2 # ground truth for the speaker s2, may not be given
        ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
        ├── FirstSpeakerID2_SecondSpeakerID2.wav
        .
        .
        .
        └── FirstSpeakerIDn_SecondSpeakerIDn.wav
```

Run the following command:

```bash
python inference.py \
    datasets=inference_custom \
    inferencer.save_path=SAVE_PATH \
    datasets.test.data_dir=TEST_DATASET_PATH \
    model=no_video_rtfs \
    inferencer.from_pretrained='saved/other/no_video_model.pth' \
    metrics.inference.0.use_pit=True \
    metrics.inference.1.use_pit=True \
    metrics.inference.2.use_pit=True \
    metrics.inference.3.use_pit=True
```

where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.
</details>

<details>
<summary>4. To separate two-speaker mixed audio using reference mouth recordings and evaluate results against ground truth
   separation, audio directory should have the following format:</summary>

```
NameOfTheDirectoryWithTestDataset
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```

Run the following command:

```bash
python inference.py \
    datasets=inference_custom \
    inferencer.save_path=SAVE_PATH \
    datasets.test.data_dir=TEST_DATASET_PATH
```

where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.
</details>

<details>
<summary>5. To evaluate the model using only predicted and ground truth audio separations, ensure that the directory containing
   them has the following format:</summary>

```
PredictDir
├── mix
│   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   .
│   .
│   .
│   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
├── s1 # prediction for the speaker s1, may not be given
│   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   .
│   .
│   .
│   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── s2 # prediction for the speaker s2
    ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
    ├── FirstSpeakerID2_SecondSpeakerID2.wav
    .
    .
    .
    └── FirstSpeakerIDn_SecondSpeakerIDn.wav

GroundTruthDir
├── mix
│   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   .
│   .
│   .
│   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
├── s1 # ground truth for the speaker s1, may not be given
│   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   .
│   .
│   .
│   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── s2 # ground truth for the speaker s2, may not be given
    ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
    ├── FirstSpeakerID2_SecondSpeakerID2.wav
    .
    .
    .
    └── FirstSpeakerIDn_SecondSpeakerIDn.wav
```

Run the following command:

```bash
python calculate_metrics.py \
    predict_dir=PREDICT_DIR_PATH \
    gt_dir=GROUND_TRUTH_DIR_PATH
```

</details>

6. Finally, if you want to reproduce results from [here](#final-results), run the following code:

```bash
python inference.py \
    inferencer.save_path=SAVE_PATH
```

Feel free to choose what kind of metrics you want to evaluate (see [this config](src/configs/metrics/inference.yaml)).

### Training

1. To reproduce AVSS model, train model with

```bash
python train.py
```

2. To reproduce audio-only SS model, train model with

```bash
python train.py \
   -cn=rtfs_pit dataloader.batch_size=10
```

It takes around 10 and 4 days to train AVSS and audio-only SS models from scratch on V100 GPU respectively.

## Final results

```angular2html
                        SI-SNRi    SDRi    PESQ    STOI    Params(M)    MACs(G)    Memory(GB)    Train time(ms)    Infer. time(ms)
RTFS-Net-12              12.95    13.33    2.42    0.92      0.771
audio-only model (R=4)   11.35    11.80    1.79    0.61      0.772
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

The pre-trained video feature extractors was taken
from [Lip-reading repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks). Thanks to the
authors for sharing their code.

## Authors

[Ilya Drobyshevskiy](https://github.com/free001style), [Boris Panfilov](https://github.com/TmBoris), [Ekaterina Grishina](https://github.com/GrishKate).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
