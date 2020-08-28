{
    "random_seed": 992020,
    "numpy_seed": 992020,
    "pytorch_seed": 992020,
    "dataset_reader" : {
        "type": "classification-pt-tsv",
        "max_tokens": 512
    },
    "train_data_path": "data/original/train.tsv",
    "validation_data_path": "data/original/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": "bert-base-cased",
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "bert-base-cased"
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 2e-5,
        }
        "num_epochs": 1,
        "cuda_device": 0,
        "use_amp": true
    }
}