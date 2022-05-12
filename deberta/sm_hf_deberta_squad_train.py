import os
# os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
# os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
# os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
# os.environ["MASTER_ADDR"]="p4dqueue-dy-p4d24xlarge-1"
# os.environ["MASTER_PORT"]="12345"
from transformers import (
    DebertaV2Config,
    DebertaV2ForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DebertaV2TokenizerFast,
    TrainerCallback
)
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import torch.distributed as dist
import smdistributed.dataparallel.torch.torch_smddp
from torch.nn.parallel import DistributedDataParallel
from datasets.filesystems import S3FileSystem


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--acc", type=int, default=1)

    # Data, model, and output directories
    # These need to be configured according to the SageMaker environment
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_EVAL"])

    args, _ = parser.parse_known_args()

    os.environ["GPU_NUM_DEVICES"] = args.n_gpus

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    # We can use the load_from_disk() API for both data in S3 as well as FSx
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # DeBERTa Question and Answering Model
    model = DebertaV2ForQuestionAnswering(DebertaV2Config.from_pretrained("microsoft/deberta-v3-base"))

    # We use a pre-trained DeBERTa tokenizer in this example
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=True,
        #disable_tqdm=True,
        evaluation_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        logging_strategy="epoch",
        gradient_accumulation_steps=args.acc,
        #adafactor = True,
        #optim="adafactor",
        gradient_checkpointing=True,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    result = trainer.train()

    # Print training metrics
    print_summary(result)
