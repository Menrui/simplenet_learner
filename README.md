# simplenet-learner

## Overview

このリポジトリは、[SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140)の実装である[DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)を参考に、hydraとpytorch-lightningを用いて実装し直したものです。  
基本的な学習機能のほか、独自実装モデルについてはonnx形式での保存をサポートしています。

## Features

- [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)と同構造のモデル、もしくはパッチ周りの処理を1×1 convolutionに置き換えた独自実装のSimpleNetの学習。
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

### トレーニングの実行

学習は`train.py`を実行することで行えます。  
configの管理には[hydra](https://hydra.cc/)を使用しているため、`config`ディレクトリ以下にあるyamlファイルを編集するか、コマンドライン引数で指定することで学習の設定を変更できます。

```bash
python train.py datamodule=mvtecad model=simplenet2d
```

学習の結果は`logs`ディレクトリ以下に保存されます。  
`logs`以下のディレクトリは学習の実行ごとに作成され、保存先の形式は`config/log_dir/`以下の設定に従います。  
学習の実行ごとに異なるディレクトリが作成されるため、過去の学習結果を上書きすることはありません。

独自のデータセットを用いる場合、以下のように`train.py`を実行してください。

```bash
python train.py datamodule=generic datamodule.category=[your dataset_directory name] model=simplenet2d
```

`[your dataset_directory name]`には、`data/`以下に配置したデータセットが格納されているディレクトリ名を指定してください。  
もし`data/`以下に独自データセット用のディレクトリを作成し、サブディレクトリにデータセットを格納する場合や、プロジェクト直下に配置されていないデータセットを使用する場合、以下のように`data_dir`を指定してください。

```bash
python train.py datamodule=generic datamodule.category=[your dataset_directory name] datamodule.data_dir=[your data_directory path] model=simplenet2d
```

`[your data_directory path]`には、データセットの親ディレクトリの絶対パスを指定してください。

もし独自データセットにtestデータが存在しない場合、以下のように`train.py`を実行してください。

```bash
python train.py test=False datamodule=generic datamodule.category=[your dataset_directory name] model=simplenet2d
```

### ONNX変換の実行

学習済みモデルをonnx形式に変換するには、`torch2onnx.py`を実行します。

```bash
python torch2onnx.py -c [your config file path] -p [your ckpt file path]
```

`[your config file path]`には、学習時に生成された`logs/`以下のディレクトリ内に格納されている`config/config.yaml`ファイルのパスを指定してください。  
`[your ckpt file path]`には、学習済みモデルのckptファイルのパスを指定してください。  
現状このコマンドは`train.py`を実行する際に`model=simplenet2d`を指定して生成されたモデルに対してのみ動作します。

コマンドの実行後、`-p`オプションで指定されたckptファイルと同名のonnxファイルがckptファイルと同じディレクトリに保存されます。

### データセットに対するモデル出力の統計情報の取得

学習済みモデルを用いてデータセットに対するモデル出力の統計情報を取得するには、`get_statistics.py`を実行します。

```bash
python get_statistics.py -c [your config file path] -p [your ckpt file path] -i [your input data directory path]
```

`[your config file path]`には、学習時に生成された`logs/`以下のディレクトリ内に格納されている`config/config.yaml`ファイルのパスを指定してください。  
`[your ckpt file path]`には、学習済みモデルのckptファイルのパスを指定してください。  
`[your input data directory path]`には、モデル出力の統計情報を取得したい画像データが入っているディレクトリのパスを指定してください。

コマンドの実行後、`-p`オプションで指定されたckptファイルと同じディレクトリに、`statistics.json`という名前のjsonファイルが保存されます。  
このjsonファイルには、入力データに対するセグメンテーションマップの平均値、分散、最大値、最小値の情報が格納されています。

## Configuration

本プロジェクトでは[hydra](https://hydra.cc/)を使用して設定管理を行っています。設定ファイルは`config`ディレクトリ以下のYAMLファイルで管理されており、コマンドライン引数でこれらの設定を上書きすることができます。

### 設定ファイル構成

```text
config/
├── train.yaml              # メインの設定ファイル
├── callbacks/              # コールバック関連の設定
│   └── default.yaml
├── datamodule/             # データモジュール関連の設定
│   ├── generic.yaml        # 独自データセット用
│   └── mvtecad.yaml        # MVTec AD用
├── log_dir/                # ログディレクトリの設定
│   ├── debug.yaml
│   └── default.yaml
├── logger/                 # ロガーの設定
│   └── csv.yaml
├── model/                  # モデル関連の設定
│   ├── simplenet.yaml
│   ├── simplenet2d.yaml
│   ├── optimizer/          # オプティマイザー設定 ※未使用
│   │   └── adam.yaml
│   └── sheduler/           # スケジューラー設定 ※未使用
│       └── steplr.yaml
└── trainer/                # トレーナー設定
    └── default.yaml
```

### 主要パラメータ一覧

#### メイン設定 (`train.yaml`)

- `data_dir`: データディレクトリのパス
- `print_config`: 設定を実行開始時に表示するかどうか
- `ignore_warnings`: 警告を無視するかどうか
- `train`: モデルの学習を行うかどうか
- `test`: テストセットでの評価を行うかどうか
- `seed`: 乱数シードの設定

#### データモジュール設定

##### 共通パラメータ (`datamodule/generic.yaml`, `datamodule/mvtecad.yaml`)

- `data_dir`: データセットのディレクトリパス
- `category`: 使用するデータセットのカテゴリ名
- `transform_cfg`: データ変換設定
  - `resize_h`, `resize_w`: リサイズ後の高さ、幅
  - `h_flip_p`, `v_flip_p`: 水平・垂直反転の確率
  - `crop_h`, `crop_w`: クロップサイズ
- `batch_size`: バッチサイズ
- `num_workers`: データローダーのワーカー数
- `pin_memory`: PyTorchのpin_memoryオプション

#### モデル設定

##### SimpleNet2D (`model/simplenet2d.yaml`)

- `anomaly_threshold`: 異常検知のしきい値
- `simplenet2d_cfg`: SimpleNetのモデル設定
  - `backborn_arch`: バックボーンアーキテクチャ（例：resnet18）
  - `backborn_pretrained`: 事前学習済みモデルを使用するかどうか
  - `backborn_trainable`: バックボーンを学習可能にするかどうか
  - `projection_channel`: 射影チャンネル数
  - `projection_layer_num`: 射影層数
  - `discriminator_layer_num`: 識別器層数
  - `discriminator_reduce_rate`: 識別器縮小率
- `projection_optimizer_cfg`: 射影ネットワークのオプティマイザー設定
  - `name`: オプティマイザー名（例：AdamW）
  - `lr`: 学習率
- `discriminator_optimizer_cfg`: 識別器のオプティマイザー設定
- `noise_generator`: ノイズ生成器の設定
  - `noise_std`: ノイズの標準偏差
  - `noise_type`: ノイズの種類
  - `noise_scaling`: ノイズのスケーリング

#### トレーナー設定 (`trainer/default.yaml`)

- `accelerator`: 使用するアクセラレータ（"cpu" または "gpu"）
- `devices`: 使用するデバイス数（-1で全デバイスを使用）
- `min_epochs`: 最小エポック数
- `max_epochs`: 最大エポック数

### コマンドライン引数による設定の上書き

hydraを用いることで、コマンドライン引数を使って設定を上書きすることができます。例えば：

```bash
# データモジュールとモデルを指定
python train.py datamodule=mvtecad model=simplenet2d

# バッチサイズを変更
python train.py datamodule.batch_size=8

# 学習率を変更
python train.py model.projection_optimizer_cfg.lr=0.0002

# 複数のパラメータを一度に変更
python train.py datamodule=generic datamodule.category=your_dataset model=simplenet2d trainer.max_epochs=30

# テスト実行せずに学習のみ実行
python train.py test=False
```

詳細な設定オプションについては、各設定ファイルを直接参照するか、hydraの[公式ドキュメント](https://hydra.cc/docs/intro/)を参照してください。
