from __future__ import annotations

import pandas as pd

from dhompo.data.urban_loader import (
    QUALITY_MISSING,
    QUALITY_OK,
    QUALITY_STALE,
    canonicalize_urban_wide,
    flag_and_fill_realtime_values,
    load_urban_config,
    preprocess_urban_wide_data,
    signal_specs,
)


def test_config_uses_urban_target_not_dhompo():
    config = load_urban_config("configs/surabaya/urban_water_level.yaml")

    assert config["target_column"] == "ketinggian_lokasi_1_hang_tuah"
    assert config["target_column"] != "Dhompo"


def test_ab_components_are_merged_to_canonical_signal():
    idx = pd.date_range("2026-01-01", periods=3, freq="30min", name="Datetime")
    raw = pd.DataFrame(
        {
            "ketinggian_lokasi_1_a_hang_tuah": [10.0, None, 30.0],
            "ketinggian_lokasi_1_b_hang_tuah": [14.0, 20.0, None],
        },
        index=idx,
    )
    config = {
        "locations": [
            {
                "key": "lokasi_1_hang_tuah",
                "name": "Hang Tuah",
                "water_level": {
                    "canonical": "ketinggian_lokasi_1_hang_tuah",
                    "components": [
                        "ketinggian_lokasi_1_a_hang_tuah",
                        "ketinggian_lokasi_1_b_hang_tuah",
                    ],
                },
            }
        ]
    }

    canonical, coverage, dropped = canonicalize_urban_wide(raw, signal_specs(config))

    assert canonical["ketinggian_lokasi_1_hang_tuah"].tolist() == [12.0, 20.0, 30.0]
    assert int(coverage.iloc[0]["non_null"]) == 3
    assert dropped == []


def test_limited_forward_fill_marks_stale_and_missing():
    idx = pd.date_range("2026-01-01", periods=4, freq="30min", name="Datetime")
    canonical = pd.DataFrame({"target": [1.0, None, None, None]}, index=idx)

    values, flags = flag_and_fill_realtime_values(canonical, max_ffill_steps=2)

    assert values["target"].tolist()[:3] == [1.0, 1.0, 1.0]
    assert pd.isna(values["target"].iloc[3])
    assert flags["target"].tolist() == [
        QUALITY_OK,
        QUALITY_STALE,
        QUALITY_STALE,
        QUALITY_MISSING,
    ]


def test_real_csv_preprocesses_with_expected_default_target():
    result = preprocess_urban_wide_data()

    assert result.target_column == "ketinggian_lokasi_1_hang_tuah"
    assert result.target_column in result.feature_columns
    assert len(result.raw) == 14122
    assert result.raw.index.freq == pd.tseries.frequencies.to_offset("30min")
    assert "Dhompo" not in result.values.columns
    assert result.coverage.set_index("column").loc[
        "ketinggian_lokasi_1_hang_tuah", "non_null"
    ] == 9990
