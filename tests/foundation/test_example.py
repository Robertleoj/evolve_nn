from project.foundation import add


def test_add() -> None:
    """Test that this function adds two numbers correctly."""
    assert add(1, 2) == 3
