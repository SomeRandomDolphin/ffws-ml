"""Regression checks for scenario paths after repository reorganization."""

import pytest

from dhompo.config import (
    PROJECT_ROOT,
    load_serving_config,
    load_yaml_config,
    resolve_artifact_path,
    resolve_path_from_config,
)
from dhompo.data.loader import _DEFAULT_DATA_PATH
from dhompo.data.urban_loader import load_urban_config
from dhompo.serving.file_predictor import _DEFAULT_MODEL_DIR as DHOMPO_MODELS
from dhompo.serving.urban_file_predictor import _DEFAULT_MODEL_DIR as SURABAYA_MODELS


@pytest.mark.parametrize("working_dir", [".", "research/dhompo", "research/surabaya"])
def test_scenario_paths_do_not_depend_on_working_directory(monkeypatch, working_dir):
    monkeypatch.chdir(PROJECT_ROOT / working_dir)
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert load_serving_config()["api"]["port"] == 8000
    assert load_urban_config()["target_column"] == "ketinggian_lokasi_1_hang_tuah"
    for scenario, name in [
        ("dhompo", "training.yaml"),
        ("surabaya", "urban_water_level.yaml"),
        ("surabaya", "urban_water_level_delta.yaml"),
    ]:
        config_path = f"configs/{scenario}/{name}"
        config = load_yaml_config(config_path)
        data_path = resolve_path_from_config(config_path, config["data_path"])
        expected_parent = PROJECT_ROOT / "data" if scenario == "dhompo" else PROJECT_ROOT / "data" / scenario
        assert data_path.parent == expected_parent
        if scenario == "dhompo":
            assert data_path.is_file()
        for source in config.get("data_sources", []):
            assert resolve_path_from_config(config_path, source["path"]).is_file()
        if "models_output_dir" in config:
            model_dir = resolve_path_from_config(config_path, config["models_output_dir"])
            assert model_dir.parent == PROJECT_ROOT / "models" / scenario
    assert _DEFAULT_DATA_PATH.is_file()
    assert DHOMPO_MODELS == PROJECT_ROOT / "models/sklearn"
    assert SURABAYA_MODELS == PROJECT_ROOT / "models/surabaya/urban"


def test_absolute_config_path_and_missing_optional_path():
    config = PROJECT_ROOT / "configs/dhompo/training.yaml"
    assert resolve_path_from_config(config, "../../data/data-clean.csv") == _DEFAULT_DATA_PATH
    assert resolve_path_from_config(config, _DEFAULT_DATA_PATH) == _DEFAULT_DATA_PATH
    assert resolve_path_from_config(config, None) is None


@pytest.mark.parametrize("recorded_path", [
    "model.pkl",
    r"Z:\old-machine\models\urban\model.pkl",
    "/old-machine/models/urban/model.pkl",
])
def test_moved_artifact_resolves_in_selected_directory(tmp_path, recorded_path):
    assert resolve_artifact_path(tmp_path, recorded_path) == tmp_path / "model.pkl"


def test_existing_absolute_artifact_keeps_its_location(tmp_path):
    artifact = tmp_path / "external.pkl"
    artifact.touch()
    assert resolve_artifact_path(tmp_path / "selected", artifact) == artifact


def test_relative_artifact_subdirectories_are_preserved(tmp_path):
    assert resolve_artifact_path(tmp_path, "h1/model.pkl") == tmp_path / "h1/model.pkl"
