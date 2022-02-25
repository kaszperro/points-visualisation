from algorithms.common import VisualizationAlgorithm


class EvaluationAlgorithm:
    def preprocess(self, visualisation_algorithm: VisualizationAlgorithm):
        raise NotImplementedError

    def calculate_from_saved_path(self, root_path):
        raise NotImplementedError

    def run(self, visualisation_algorithm: VisualizationAlgorithm):
        save_path = self.preprocess(visualisation_algorithm)
        return self.calculate_from_saved_path(save_path)
