# simplenet-learner

## Overview

このリポジトリは、[SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140)の実装である[DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)を参考に、hydraとpytorch-lightningを用いて実装し直したものです。  
基本的な学習機能のほか、独自実装モデルについてはonnx形式での保存をサポートしています。

## Features

- [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)と同構造のモデル、もしくはパッチ周りの処理を1 by 1 convに置き換えた独自実装のSimpleNetの学習。
- 学習済みモデルを用いた推論及び精度の計測。
- 学習済みモデルを用いた単画像に対する推論。
- 学習済みモデルをonnx形式で保存。
- データセットに対するモデル出力の統計情報の取得。
- 独自データセットの適用。

## Requirements

```markdown
- python>=3.11
- colorlog>=6.9.0
- hydra-colorlog>=1.2.0
- hydra-core>=1.3.2
- lightning>=2.5.0.post0
- rich>=13.9.4
- scikit-learn>=1.6.1
- torch==2.1.2+cu118
- torchinfo>=1.8.0
- torchmetrics>=1.6.0
- torchvision==0.16.2+cu118
- tqdm>=4.67.1
- numpy<2
- matplotlib>=3.10.0
- onnx>=1.17.0
- pycryptodome>=3.21.0
- python-dotenv>=1.0.1
```

## Installation

### Using pip

開発者からwheelファイルを入手してください。

```bash
pip install simplenet_learner-0.5.0-py3-none-any.whl
```

### Using uv

[uv - Get Started](https://docs.astral.sh/uv/#getting-started)

```bash
uv sync
```

## Usage

### 画像データの配置規則

データセットは以下のようなディレクトリ構造で配置してください。  
独自のデータセットを用いる場合は`data/your_dataset`以下の構成になるように、mvtecadデータセットを用いる場合は`data/mvtecad`以下の構成になるように配置してください。  
独自のデータセットを用いる場合、`train/good`ディレクトリが存在すれば`test`ディレクトリは存在しなくてもtrain.pyを実行可能です。その場合、テストデータを用いた精度の検証は行えません。

```text
/
├─ config/
├─ data/
│  ├─ your_dataset
│  │  ├─ train
│  │  │  └─ good
│  │  └─ test
│  │     ├─ good
│  │     └─ bad
│  └─ mvtecad
│     ├─ bottle
│     │  ├─ ground_truth
│     │  ├─ train
│     │  │  └─ good
│     │  └─ test
│     │     ├─ good
│     │     └─ bad
│     └─ ...
├─ logs/
├─ notebooks/
├─ scripts/
├─ src/
└─ ...
```
