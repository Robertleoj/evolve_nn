from project.utils.example import example_util


def test_example_util() -> None:
    """Test that the example util works correctly."""
    assert example_util(1, 2) == 3
