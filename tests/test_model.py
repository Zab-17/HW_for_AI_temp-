from training.model import build_baseline_model


def test_baseline_has_the_parameter_counts_calculated_by_hand():
    model = build_baseline_model()

    parameters_by_layer = {layer.name: layer.count_params() for layer in model.layers if layer.count_params()}

    assert parameters_by_layer == {"conv1": 104, "conv2": 528, "dense1": 3600, "dense2": 68}
    assert model.count_params() == 4300


def test_baseline_outputs_one_probability_per_class():
    assert build_baseline_model().output_shape == (None, 4)
