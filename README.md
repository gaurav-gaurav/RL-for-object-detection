# Active Perception for Visual Signal Recovery

Code for the ICASSP 2023 paper on learning *where to look*: a reinforcement
learning agent corrects a degraded visual signal before it reaches a frozen
object detector, improving detection under noise and occlusion.

> Gaurav Chaudhary, Laxmidhar Behera, Tushar Sandhan.
> *Active Perception System for Enhanced Visual Signal Recovery Using Deep
> Reinforcement Learning.* IEEE ICASSP, Rhodes, Greece, 2023.

## Method

A pretrained YOLOv3 detector is held fixed. A PPO agent sits in front of it and
learns a correction to apply to the incoming image — the reward is the
detector's own performance, so the agent is optimised directly against
downstream detection quality rather than a hand-designed image metric.

| File | Role |
|---|---|
| `main.py` | Training entry point (single-image rollouts). |
| `main_batch.py` | Batched variant of the training loop. |
| `Eval.py` | Evaluation and reporting. |
| `PPO.py` | PPO update. |
| `policy.py` | Policy and value networks, including the attention module. |
| `resnet_policy.py` | ResNet-backbone policy variant. |
| `ICM_Policy.py` | Intrinsic curiosity variant of the policy. |
| `yolo_detect.py` | Wraps the frozen detector as an environment. |
| `get_yolo_output.py` | Pulls raw detections for reward computation. |
| `resize_voc.py` | Rescales PASCAL VOC annotations to the working resolution. |
| `utils.py` | Project helpers (metrics, plotting, bookkeeping). |

## Setup

```bash
git clone https://github.com/gaurav-gaurav/RL-for-object-detection.git
cd RL-for-object-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two things are **not** in the repository and must be fetched separately.

**YOLOv3 weights and config.** The code expects:

```
cfg/yolov3.cfg          cfg/yolov3-voc.cfg
data/coco.names         data/voc.names
Weights/yolov3.weights  Weights/yolov3-voc_final.weights
```

Config and class-name files come from
[pjreddie/darknet](https://github.com/pjreddie/darknet); weights from
[pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights).

**PASCAL VOC images.** Expected under `data/voc` (override with
`--path_of_dataset`). The annotation CSVs used by the experiments
(`labels.csv`, `resized_labels.csv`) *are* included; `resize_voc.py`
regenerates them for a different input resolution.

## Usage

```bash
# Train
python main.py --dataset voc --path_of_dataset data/voc --epochs 400 --seed 123

# Batched training loop
python main_batch.py --dataset voc --batch_size 64

# Evaluate a trained policy
python Eval.py
```

### Selected arguments

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `voc` | Dataset name |
| `--path_of_dataset` | `data/voc` | Image root |
| `--image_dim` | `(128, 128)` | Working resolution |
| `--epochs` / `--steps` | `400` / `2000` | Training length |
| `--batch_size` | `64` | Batch size |
| `--lr` / `--gamma` / `--lambda` | `1e-3` / `0.99` / `0.95` | PPO hyperparameters |
| `--attention` | `False` | Enable the attention module in the policy |
| `--action_type` | `discrete` | `discrete` or continuous correction actions |
| `--noise_type` / `--noise_scale` | `1` / `full` | Degradation applied to the input |
| `--confidence` / `--nms_thresh` | `0.5` / `0.3` | Detector thresholds |
| `--iou_threshold` | `0.5` | IoU threshold for scoring |

`python main.py --help` lists everything.

## Attribution

`darknet.py`, `util.py`, `bbox.py`, `preprocess.py` and `pallete` are adapted
from [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
and provide the frozen detector the agent acts upon. They are not part of this
paper's contribution; each carries a header noting its origin. Everything else
is the authors' own.

Note that `util.py` (detector utilities, third-party) and `utils.py` (project
helpers, ours) are different files — an unfortunate naming collision kept for
compatibility with the original code.

## Scope and status

Research code released to support the paper, not a maintained library. It is
the code used for the experiments, tidied for release: scratch scripts and an
unused test asset removed, dead code stripped, and two duplicate modules that
differed only by capitalisation (`Policy.py`, `Yolo_detect.py`) deleted in
favour of the versions actually imported — those duplicates silently broke
clones on macOS and Windows. No algorithmic changes were made.

## Citation

```bibtex
@inproceedings{chaudhary2023active,
  title     = {Active Perception System for Enhanced Visual Signal Recovery Using Deep Reinforcement Learning},
  author    = {Chaudhary, Gaurav and Behera, Laxmidhar and Sandhan, Tushar},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1--5},
  year      = {2023}
}
```

## License

MIT for our code — see [LICENSE](LICENSE), which also records the third-party
portions.
