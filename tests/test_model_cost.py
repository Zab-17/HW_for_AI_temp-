from pathlib import Path

import pytest

from analysis.model_cost import operator_costs, size_breakdown
from analysis.tflite_model import TfliteModel

MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "fp32" / "model_fp32.tflite"
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run training.convert_fp32 first")


@needs_model
def test_costs_match_the_hand_calculation():
    costs = operator_costs(TfliteModel.from_file(MODEL_PATH))

    by_operator = {(c.index, c.operator): (c.parameters, c.multiply_accumulates) for c in costs}

    assert by_operator[(0, "CONV_2D")] == (104, 36_864)
    assert by_operator[(2, "CONV_2D")] == (528, 21_504)
    assert by_operator[(5, "FULLY_CONNECTED")] == (3_600, 3_584)
    assert by_operator[(6, "FULLY_CONNECTED")] == (68, 64)
    assert sum(c.parameters for c in costs) == 4_300
    assert sum(c.multiply_accumulates for c in costs) == 62_016


@needs_model
def test_size_breakdown_adds_up_to_the_file_size():
    model = TfliteModel.from_file(MODEL_PATH)

    parts = size_breakdown(model)

    assert parts["float32 parameters"] == 17_200
    assert sum(parts.values()) == model.size_bytes
