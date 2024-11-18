<h1 align="center">Audio-Visual Source Separation(AVSS)</h1>

<p align="center">
  <a href="#about">About</a> •
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

[//]: # (See [wandb report]&#40;https://wandb.ai/free001style/ASR/reports/Report-of-ASR--Vmlldzo5NDc2NDAw&#41; with all experiments.)

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

1) To separate two-speaker mixed audio, your audio directory should have the following format:
```bash
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

<details>
<summary>Run the following command:</summary>

```bash
python inference.py \
    datasets=inference_custom \
    datasets.test.data_dir=TEST_DATASET_PATH \
    inferencer.save_path=SAVE_PATH \
    model.use_video=False \
    model.channel_dim=392 \
    model.R=4 \
    inferencer.from_pretrained='saved/no_video/no_video_model.pth'
```
   where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.

</details>

2) To separate two-speaker mixed audio using reference mouth recordings, your audio and video directory should have the following format:
```bash
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

<details>
<summary>Run the following command:</summary>

```bash
python inference.py \
    datasets=inference_custom \
    inferencer.save_path=SAVE_PATH \
    datasets.test.data_dir=TEST_DATASET_PATH \
    inferencer.from_pretrained='saved/R4/model_best.pth'
```
   where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.

</details>

3) To separate two-speaker mixed audio and evaluate results against ground truth separation, your audio directory should have the following format:

```bash
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
<details>
<summary>Run the following command:</summary>

```bash
!python inference.py \
    datasets=inference_custom \
    inferencer.save_path='tmp' \
    datasets.test.data_dir='data/custom' \
    model=no_video_rtfs \
    inferencer.from_pretrained='saved/no_video/no_video_model.pth' \
    metrics.inference.0.use_pit=True \
    metrics.inference.1.use_pit=True \
    metrics.inference.2.use_pit=True \
    metrics.inference.3.use_pit=True
```
where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.
</details>

4) To separate two-speaker mixed audio using reference mouth recordings and evaluate results against ground truth separation, your audio directory should have the following format:
```bash
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
<details>
<summary>Run the following command:</summary>

```bash
!python inference.py \
    datasets=inference_custom \
    inferencer.save_path=SAVE_PATH \
    datasets.test.data_dir=TEST_DATASET_PATH \
    inferencer.from_pretrained='saved/R4/model_best.pth'
```
where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with test data.
</details>

5)  To evaluate the model using only predicted and ground truth audio separations, ensure that the directory containing them has the following format:
```bash
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
<details>
<summary>Run the following command:</summary>

```bash
!python calculate_metrics.py \
    predict_dir=PREDICT_DIR_PATH \
    gt_dir=GROUND_TRUTH_DIR_PATH
```
</details>

6) Finally, if you want to reproduce results from [here](#final-results), run the following code:
```bash
python inference.py \
    inferencer.from_pretrained='saved/R4/model_best.pth'
```
   Feel free to choose what kind of metrics you want to evaluate (
   see [this config](src/configs/metrics/inference.yaml)).

### Training

The model training contains of 3 stages. To reproduce results, train model using the following commands:

1. To reproduce AVSS model, train model with

   ```bash
   TODO
   python train.py writer.run_name="part1" dataloader.batch_size=230 transforms=example_only_instance trainer.early_stop=47
   ```

2. To reproduce ASS model, train model with

   ```bash
   TODO
   python train.py writer.run_name="part2" dataloader.batch_size=230 trainer.resume_from=part1/model_best.pth datasets.val.part=test-other
   ```

It takes around 10 and 4 days to train AVSS and ASS models from scratch on V100 GPU respectively.


## Final results


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## Authors

[Ilya Drobyshevskiy](https://github.com/free001style), [Boris Panfilov](https://github.com/TmBoris), [Ekaterina Grishina](https://github.com/GrishKate).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
