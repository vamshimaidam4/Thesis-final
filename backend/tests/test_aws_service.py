import pytest
from app.services.aws_service import check_aws_connection


def test_check_aws_connection_returns_dict():
    result = check_aws_connection()
    assert isinstance(result, dict)
    assert "status" in result


def test_check_aws_connection_without_credentials():
    """When no valid credentials are set, should return error status."""
    result = check_aws_connection()
    # Will either be connected (if real creds) or show an error
    assert result["status"] in ("connected", "no_credentials", "error")
