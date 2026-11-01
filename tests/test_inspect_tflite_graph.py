import pytest

from analysis.inspect_tflite_graph import _camel_to_operator_name


@pytest.mark.parametrize(
    "camel, operator",
    [
        ("Conv2D", "CONV_2D"),
        ("MaxPool2D", "MAX_POOL_2D"),
        ("FullyConnected", "FULLY_CONNECTED"),
        ("Softmax", "SOFTMAX"),
    ],
)
def test_resolver_method_names_map_to_operator_names(camel, operator):
    assert _camel_to_operator_name(camel) == operator
