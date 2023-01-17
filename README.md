# DeBERTa_HuggingFace_SageMakerDDP

We will look at the SageMaker Distributed Data Parallel (SMDDP) Library and how easy it is to integrate it with open-source libraries like HuggingFace Transformers to train huge NLP models with no changes to your training script. More specifically, we will train a DeBERTaV3 model on the SQUAD dataset for the question-answering task. We will also see how SMDDP attains a higher, near-linear scaling efficiency compared to existing data parallel libraries like the native PyTorch DistributedDataParallel library which uses NCCL as the backend.
