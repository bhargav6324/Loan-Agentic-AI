import pytest
import json
import httpx
from unittest.mock import patch, Mock
from config import settings
from agents.ollama_client import (
    check_health, generate_text, generate_json,
    OllamaError, OllamaConnectionError, OllamaJSONError
)

@patch("httpx.get")
def test_check_health_success(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": settings.OLLAMA_MODEL}]}
    mock_get.return_value = mock_response
    
    assert check_health() is True

@patch("httpx.get")
def test_check_health_missing_model(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "some-other-model"}]}
    mock_get.return_value = mock_response
    
    with pytest.raises(OllamaConnectionError, match="not available"):
        check_health()

@patch("httpx.get")
def test_check_health_connection_failure(mock_get):
    mock_get.side_effect = httpx.RequestError("Connection failed")
    with pytest.raises(OllamaConnectionError, match="Could not connect"):
        check_health()

@patch("httpx.post")
def test_generate_text_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Hello world"}
    mock_post.return_value = mock_response
    
    res = generate_text("Say hi")
    assert res == "Hello world"

@patch("httpx.post")
def test_generate_json_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": '{"monthly_income": 50000}'}
    mock_post.return_value = mock_response
    
    res = generate_json("Extract income")
    assert res == {"monthly_income": 50000}

@patch("httpx.post")
def test_generate_json_malformed(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": '{monthly_income: 50000'}
    mock_post.return_value = mock_response
    
    with pytest.raises(OllamaJSONError, match="malformed JSON"):
        generate_json("Extract income")

@patch("httpx.post")
def test_generate_json_not_object(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": '["array", "not", "object"]'}
    mock_post.return_value = mock_response
    
    with pytest.raises(OllamaJSONError, match="not a JSON object"):
        generate_json("Extract array")

@patch("httpx.post")
def test_generate_timeout(mock_post):
    mock_post.side_effect = httpx.TimeoutException("Timeout")
    with pytest.raises(OllamaConnectionError, match="Request to Ollama failed"):
        generate_text("Say hi")

@patch("httpx.post")
def test_generate_missing_response(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"other_field": "test"}
    mock_post.return_value = mock_response
    
    with pytest.raises(OllamaError, match="Missing 'response' field"):
        generate_text("Say hi")
