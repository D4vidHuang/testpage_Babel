"""Orchestration logic for running evaluations across configurations."""

from typing import List, Dict, Any

from .config import JudgeConfig


class JudgeManager:
    """Manages multiple judging configurations and evaluates text with them."""

    def __init__(self, configs: List[JudgeConfig]):
        self.configs = configs

    def run(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Run text through all judge configurations and models.

        Returns:
            Nested dict: {config_name: {model_name: raw_response}}
        """
        results: Dict[str, Dict[str, Any]] = {}

        for cfg in self.configs:
            judge = cfg.create_judge()
            cfg_results = {}

            for i, model_name in enumerate(cfg.models):
                cfg_results[model_name] = judge.judge(text, model_index=i)

            results[cfg.name] = cfg_results

        return results

    def run_dataset(self, dataset: List[str]) -> Dict[str, Dict[str, List[Any]]]:
        """Evaluate a dataset of text items through all configurations.

        Returns:
            Nested dict: {config_name: {model_name: [response1, response2, ...]}}
        """
        results: Dict[str, Dict[str, List[Any]]] = {}

        for cfg in self.configs:
            judge = cfg.create_judge()
            cfg_results: Dict[str, List[Any]] = {model: [] for model in cfg.models}

            for snippet in dataset:
                for i, model_name in enumerate(cfg.models):
                    cfg_results[model_name].append(judge.judge(snippet, model_index=i))

            results[cfg.name] = cfg_results

        return results
