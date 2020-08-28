from classifier.dataset_readers.dataset_reader import ClassificationTsvReader
from classifier.dataset_readers.dataset_reader_pt import ClassificationPtTsvReader
from allennlp.common.util import ensure_list

def test_rey_reader_1(project_root_dir_path, test_fixtures_dir_path, test_log):

    data_file_path = test_fixtures_dir_path / 'data' / 'train_500.tsv'
    reader = ClassificationTsvReader()
    instances = ensure_list(reader.read(str(data_file_path)))
    print(instances)

    assert len(instances) == 10
    # instances[0].fields["text"].tokens
    assert instances[0].fields["label"].label == '27'


def test_rey_reader_2(project_root_dir_path, test_fixtures_dir_path, test_log):

    data_file_path = test_fixtures_dir_path / 'data' / 'train_500.tsv'
    reader = ClassificationPtTsvReader()
    instances = ensure_list(reader.read(str(data_file_path)))
    print(instances)

    assert len(instances) == 10
    print(instances[0].fields["text"].tokens)
    assert instances[0].fields["label"].label == '27'