from allennlp.common.testing import ModelTestCase

class TestModel(ModelTestCase):

    def test_model_1(self, project_root_dir_path, test_fixtures_dir_path, test_log):
        param_file = str(test_fixtures_dir_path / "models" / "allennlp_goemotions_config.jsonnet")
        self.ensure_model_can_train_save_and_load(param_file)