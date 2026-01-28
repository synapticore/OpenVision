# OpenVision Coding Agent Instructions

## Repository Overview

**OpenVision** is a JAX/Flax vision-language training framework for Google Cloud TPU. Two architectures: (1) Original: CLIP + captioning, (2) OpenVision 2: caption-only (1.5-2× faster, 1.8× less memory).

**Stack**: Python 3.10+, JAX 0.4.38, Flax 0.10.2, TensorFlow 2.18.0 (data), PyTorch CPU (conversion only) | **Size**: 14MB, 77 .py files | **Target**: TPU v4 pods, ViT models 5M-1B params

## Critical Setup Requirements

### Environment Setup (ALWAYS REQUIRED)

**⚠️ CRITICAL**: This project REQUIRES Python 3.10 and a virtual environment. System Python will fail.

```bash
# ALWAYS run these steps IN ORDER on fresh TPU VM or local machine:
sudo apt update && sudo apt install -y python3.10-venv
python3.10 -m venv ~/openvision/openvision_env
source ~/openvision/openvision_env/bin/activate  # OR: . ~/openvision/openvision_env/bin/activate

# Install dependencies (this takes ~5-10 minutes)
bash setup.sh DEVICE=tpu JAX_VERSION=0.4.38
# For GPU: bash setup.sh DEVICE=gpu JAX_VERSION=0.4.38
```

**Common Errors & Fixes**:
- **Error**: "No module named 'jax'" → Virtual environment not activated or setup.sh not run
- **Error**: "apt_pkg module not found" → Use Python 3.10, not system Python
- **Timing**: setup.sh takes 5-10 minutes. JAX installation is slowest part.
- **Ordering**: ALWAYS activate venv BEFORE running setup.sh

**Dependencies**: 40+ packages in requirements.txt (JAX, TensorFlow, Transformers, WandB). PyTorch is CPU-only. Custom libtpu.so via LIBTPU_GCS_PATH optional.

## Project Structure

**Root Files**: README.md (docs+model zoo), LICENSE (Apache 2.0), .gitignore (excludes venv/npz/logs), requirements.txt, setup.sh (env setup), test.py (PyTorch test), tpu_command.sh (TPU mgmt), assets/ (vocab/logos), scripts/ (training), src/ (code)

**src/ Structure**:
- **Entry points**: main_clip.py (original), main_openvision2.py (v2)
- **configs/**: openvision.py, openvision2.py, common.py (ML Collections configs)
- **models/**: vit.py (ViT), text_transformer.py (encoder), text_decoder*.py (decoders), two_towers.py
- **datasets/**: input_pipeline.py (TFRecord), build_transforms.py (augmentation), tfds.py
- **evaluators/**: common.py, proj/image_text/{discriminative_classifier,retrieval,image_text_retrieval}.py (ImageNet/COCO/Flickr evals)
- **losses/**: common.py (contrastive, captioning)
- **optim/**: build_optax.py (AdamW, warmup, cosine)
- **helpers/**: utils.py (ckpt I/O, metrics), sharding.py (JAX mesh), registry.py
- **convert_upload/**: transfer_jax2hf.py (JAX→PyTorch), open_clip/ (custom fork)

**scripts/project/**: openvision/train.sh, openvision2/train.sh - multi-stage training (84px→224px→384px), execute via gcloud ssh

## Training Workflow

**BEFORE TRAINING**: Edit train.sh to set: (1) TPU: ZONE, PROJECT_ID, TPU_NAME; (2) Data: IN1K_DATA_DIR, COCO_DATA_DIR, Flickr_DATA_DIR, DATACOMP_PATH (all GCS paths); (3) Tracking: EXP_PATH, WANDB_PROJ, WANDB_ENTITY; (4) Model: MODEL (Ti/16,S/16,B/16,L/14,H/14,g/14), DECODER_NAME; (5) Hyperparams: BATCH_FACTOR, PRE_LR, FT_LR, PRE_RES, FT_RES

**Training Commands** (run on TPU VM via gcloud ssh):
```bash
# OpenVision: python3 -m src.main_clip --config=src/configs/openvision.py:res=84,img=L/14,... --workdir=gs://bucket/exp
# OpenVision 2: python3 -m src.main_openvision2 --config=src/configs/openvision2.py:res=84,img=L/14,keep_ratio=0.35,... --workdir=gs://bucket/exp
```
**Stages**: (1) Pre-train low-res (84px, 10k epochs~days-weeks), (2) Fine-tune mid-res (224px, 800 epochs~hours-days), (3) Optional high-res (336/384px, 200 epochs). Always runs on TPU VM in tmux/screen. Checkpoints auto-save to --workdir (GCS).

**Data Format**: Pre-training: TFRecord shards {'image/encoded', 'txt', 'llava_caption'}, default Recap-DataComp-1B. Eval: ImageNet (TFDS), COCO/Flickr30k (TFRecord). Set paths: DATACOMP_PATH, IN1K_DATA_DIR, COCO_DATA_DIR, Flickr_DATA_DIR (all gs:// paths in train.sh).

**Tests**: test.py (PyTorch inference, requires torch). Validation: (1) Syntax: `python3 -m py_compile src/main_*.py`, (2) Config: run with --config.eval_only=True, (3) Data: check first batch loads. Common errors: wrong GCS paths→auth with gcloud, OOM→reduce BATCH_FACTOR, WandB fails→wandb login or disable.

**TPU VM Tools**: tpu_command.sh interactive menu (`. ./tpu_command.sh && tpu`) - functions: ssh_vm, sync_dirs (rsync code), kill_jobs, prepare_env, check (lsof), rm_tpu_logs. Workflow: sync_dirs→ssh_vm→tmux→run train→detach. Manual: `gcloud alpha compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=all --command="CMD"`

**JAX→PyTorch Conversion**: Edit scripts/convert_ckpt/script_tiny.sh. `export JAX_PLATFORMS='cpu' && python3 -m src.convert_upload.transfer_jax2hf --config=src/configs/openvision.py:... --workdir=gs://ckpt --config.hf_upload.repo_name=user/model --config.hf_upload.token=hf_TOKEN`

**Config Overrides**: `--config=path.py:key1=val1,key2=val2` or `--config.model.image.dtype=bfloat16`. Key params: img (arch), txt_name, txt_decoder_name, base_lr, imagenet_epoch, batch_factor, res, token_len, mask_ratio, data_parallelism, fsdp_parallelism, tensor_parallelism, ft_from (ckpt path).

**Known Issues**: (1) utils.py checkpoint key quirk (opt~1~0~0), (2) mask_ratio only 0/0.25/0.35, (3) resolution interpolation TODO, (4) TPU lockfile→`sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs`, (5) "TPU in use"→kill_jobs, (6) First JAX compile 5-10min.

**CI/CD**: NO GitHub Actions workflows. Manual checks: `pylint src/ --disable=all --enable=E`, `pyink src/`, `pre-commit run --all-files` (if config exists).

**Best Practices**: (1) Always activate venv, (2) Validate GCS paths exist (gsutil ls), (3) Test with imagenet_epoch=10 first, (4) Monitor WandB after 100 steps, (5) Use tmux on TPU, (6) Clean with rm_tpu_logs if device locked, (7) Checkpoint is checkpoint.npz for ft_from.

**Quick Commands**:
```bash
# Setup: python3.10 -m venv ~/openvision/openvision_env && . ~/openvision/openvision_env/bin/activate && bash setup.sh DEVICE=tpu JAX_VERSION=0.4.38
# Train: bash scripts/project/openvision2/train.sh  # Edit script first!
# TPU: . ./tpu_command.sh && tpu
# Validate: python3 -m src.main_openvision2 --config=src/configs/openvision2.py:res=224 --config.eval_only=True --workdir=/tmp/test
```

**⚠️ TRUST THESE INSTRUCTIONS**: Validated in actual repo environment. Only search if: (1) instructions incomplete, (2) undocumented errors, (3) need implementation details. For routine tasks, follow exactly.
