{
    "dataset_reader" : {
        "type": "classification-pt-tsv",
    },
    "train_data_path": "test_fixtures/data/train_10.tsv",
    "validation_data_path": "test_fixtures/data/validation_5.tsv",
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
        "optimizer": "adam",
        "num_epochs": 1
    }
}