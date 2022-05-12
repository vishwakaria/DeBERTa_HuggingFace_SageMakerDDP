ARG region

# Download base PT DLC. Note that this notebook requires a HF DLC with >= PT 1.10.2
FROM 763104351884.dkr.ecr.${region}.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04

ARG WORK_DIR="transformers_build"
WORKDIR $WORK_DIR
RUN pwd; pip install git+https://github.com/huggingface/transformers; echo "installed tran"; cd transformers; \
    python setup.py; \
    cd ../..; rm -rf $WORK_DIR;

