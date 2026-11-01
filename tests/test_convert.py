import numpy as np

from training.convert_fp32 import as_c_array, compare_outputs, with_fixed_batch
from training.model import build_baseline_model


def test_fixed_batch_copy_has_the_same_weights_and_outputs():
    model = build_baseline_model()
    window = np.random.default_rng(0).normal(size=(1, 128, 3, 1)).astype(np.float32)

    fixed = with_fixed_batch(model, batch_size=1)

    assert fixed.input_shape == (1, 128, 3, 1)
    assert [layer.name for layer in fixed.layers] == [layer.name for layer in model.layers]
    np.testing.assert_array_equal(fixed(window), model(window))


def test_c_array_lists_every_byte_and_the_length():
    source = as_c_array(bytes([1, 2, 255]), "g_model")

    assert "0x01, 0x02, 0xff" in source
    assert "const unsigned int g_model_len = 3;" in source


def test_compare_outputs_counts_agreement_and_accuracy():
    reference = np.array([[0.9, 0.1], [0.2, 0.8]])
    candidate = np.array([[0.9, 0.1], [0.6, 0.4]])

    report = compare_outputs(reference, candidate, labels=np.array([0, 1]))

    assert report["same_prediction"] == 1
    assert report["reference_accuracy"] == 1.0
    assert report["candidate_accuracy"] == 0.5
    assert np.isclose(report["max_abs_difference"], 0.4)
