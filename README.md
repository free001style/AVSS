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
NameOfTheDirectoryWithUtterances
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
   TODO
   python inference.py \
    datasets=inference_custom \
    datasets.test.data_dir=TEST_DATASET_PATH \
    inferencer.save_path=SAVE_PATH \
    model.use_video=False \
    model.channel_dim=392 \
    model.R=4 \
    inferencer.from_pretrained='saved/no_video/no_video_model.pth'
   ```
   where `SAVE_PATH` is a path to save separation predictions and `TEST_DATASET_PATH` is directory with source audio.
2) To separate two-speaker mixed audio using reference mouth recordings, your audio and video directory should have the following format:
```bash
NameOfTheDirectoryWithUtterances
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
   TODO
   ```
   where `SAVE_PATH` is a path to save predicted text and `TEST_DATA` is directory with audio.
3) To separate two-speaker mixed audio and evaluate results against ground truth separation, your audio directory should have the following format:
```bash
NameOfTheDirectoryWithUtterances
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
   Then run the following command:
   ```bash
   TODO
   ```
4) To separate two-speaker mixed audio using reference mouth recordings and evaluate results against ground truth separation, your audio directory should have the following format:
```bash
NameOfTheDirectoryWithUtterances
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
   Then run the following command:
   ```bash
   TODO
   ```
5)  To evaluate the model using only predicted and ground truth audio separations, ensure that the directory containing them has the following format:
   ```
   NameOfTheDirectoryWithUtterances
    ├── ID1.json # may be flac or mp3
    .
    .
    .
    └── IDn.json

   ID1 = {"pred_text": "ye are newcomers", "text": "YE ARE NEWCOMERS"}
   ```
   Then run the following command:
   ```bash
   TODO
   python calculate_wer_cer.py --dir_path=DIR
   ```
6) Finally, if you want to reproduce results from [here](#final-results), run the following code:
   ```bash
   TODO
   python inference.py dataloader.batch_size=500 inferencer.save_path=SAVE_PATH datasets.test.part="test-other"
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
