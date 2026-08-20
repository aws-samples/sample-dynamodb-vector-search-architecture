"""Unit tests for the action group Lambda function."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing the module
os.environ["TABLE_NAME"] = "test-table"
os.environ["INDEX_NAME"] = "content-embedding-index"
os.environ["EMBEDDING_MODEL_ID"] = "amazon.titan-embed-text-v2:0"
os.environ["EMBEDDING_DIMENSIONS"] = "1024"


def make_streaming_body(data: dict) -> MagicMock:
    """Create a mock StreamingBody that returns JSON data on read()."""
    body = MagicMock()
    body.read.return_value = json.dumps(data).encode("utf-8")
    return body


def make_fake_embedding(dimensions: int = 1024) -> list:
    """Generate a fake embedding vector."""
    return [0.01 * i for i in range(dimensions)]


def build_agent_event(function_name: str, parameters: dict, action_group: str = "knowledge-operations") -> dict:
    """Build a Bedrock agent action group invocation event.

    Args:
        function_name: The function to invoke.
        parameters: Dict of parameter name -> value.
        action_group: The action group name.
    """
    return {
        "actionGroup": action_group,
        "function": function_name,
        "parameters": [
            {"name": name, "value": value}
            for name, value in parameters.items()
        ],
    }


@patch("boto3.client")
def test_semantic_search_routing(mock_boto_client):
    """semantic_search function routes to DynamoDB search_vectors."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    fake_embedding = make_fake_embedding()
    mock_bedrock.invoke_model.return_value = {
        "body": make_streaming_body({"embedding": fake_embedding})
    }
    mock_dynamodb.search_vectors.return_value = {
        "Items": [
            {
                "entity_id": {"S": "DOC-001"},
                "title": {"S": "Retry strategy guide"},
                "category": {"S": "runbook"},
                "entity_type": {"S": "document"},
            }
        ]
    }

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    event = build_agent_event(
        function_name="semantic_search",
        parameters={"query": "retry strategy", "category": "runbook"},
    )

    result = module.handler(event, None)

    mock_dynamodb.search_vectors.assert_called_once()
    call_kwargs = mock_dynamodb.search_vectors.call_args[1]
    assert call_kwargs["TableName"] == "test-table"
    assert call_kwargs["IndexName"] == "content-embedding-index"


@patch("boto3.client")
def test_get_item_details_routing(mock_boto_client):
    """get_item_details function routes to DynamoDB get_item."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    mock_dynamodb.get_item.return_value = {
        "Item": {
            "entity_id": {"S": "DOC-001"},
            "sk": {"S": "METADATA"},
            "title": {"S": "Retry strategy guide"},
            "content": {"S": "Document content here."},
            "category": {"S": "runbook"},
        }
    }

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    event = build_agent_event(
        function_name="get_item_details",
        parameters={"entity_id": "DOC-001"},
    )

    result = module.handler(event, None)

    mock_dynamodb.get_item.assert_called_once()
    call_kwargs = mock_dynamodb.get_item.call_args[1]
    assert call_kwargs["TableName"] == "test-table"
    assert call_kwargs["Key"] == {
        "entity_id": {"S": "DOC-001"},
        "sk": {"S": "METADATA"},
    }


@patch("boto3.client")
def test_unknown_function_returns_error(mock_boto_client):
    """An unknown function name returns an error in the response body."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    event = build_agent_event(
        function_name="unknown_func",
        parameters={"some_param": "some_value"},
    )

    result = module.handler(event, None)

    body = json.loads(result["response"]["functionResponse"]["responseBody"]["TEXT"]["body"])
    assert "error" in body
    assert "unknown_func" in body["error"]


@patch("boto3.client")
def test_response_format_matches_bedrock_structure(mock_boto_client):
    """Verify response has the required Bedrock agent response structure."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    mock_dynamodb.get_item.return_value = {
        "Item": {
            "entity_id": {"S": "DOC-001"},
            "sk": {"S": "METADATA"},
            "title": {"S": "Test doc"},
            "content": {"S": "Content here."},
        }
    }

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    event = build_agent_event(
        function_name="get_item_details",
        parameters={"entity_id": "DOC-001"},
        action_group="knowledge-operations",
    )

    result = module.handler(event, None)

    # Verify top-level structure
    assert result["messageVersion"] == "1.0"
    assert "response" in result
    assert result["response"]["actionGroup"] == "knowledge-operations"
    assert result["response"]["function"] == "get_item_details"
    assert "functionResponse" in result["response"]
    assert "responseBody" in result["response"]["functionResponse"]
    assert "TEXT" in result["response"]["functionResponse"]["responseBody"]
    assert "body" in result["response"]["functionResponse"]["responseBody"]["TEXT"]

    # Verify body is valid JSON
    body = json.loads(result["response"]["functionResponse"]["responseBody"]["TEXT"]["body"])
    assert isinstance(body, dict)


@patch("boto3.client")
def test_generate_embedding_returns_list_of_floats(mock_boto_client):
    """generate_embedding returns a list of floats from Bedrock response."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    fake_embedding = make_fake_embedding(1024)
    mock_bedrock.invoke_model.return_value = {
        "body": make_streaming_body({"embedding": fake_embedding})
    }

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    result = module.generate_embedding("test query text")

    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(v, float) for v in result)


@patch("boto3.client")
def test_topk_capped_at_100(mock_boto_client):
    """semantic_search caps TopK at 100 even when max_results is higher."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    fake_embedding = make_fake_embedding()
    mock_bedrock.invoke_model.return_value = {
        "body": make_streaming_body({"embedding": fake_embedding})
    }
    mock_dynamodb.search_vectors.return_value = {"Items": []}

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    module.semantic_search("test query", "runbook", max_results=200)

    call_kwargs = mock_dynamodb.search_vectors.call_args[1]
    assert call_kwargs["TopK"] == 100


@patch("boto3.client")
def test_get_item_details_not_found_returns_error(mock_boto_client):
    """get_item_details returns an error when the item doesn't exist."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    # Return empty response (no Item key)
    mock_dynamodb.get_item.return_value = {}

    import importlib
    import lambda_functions.action_group.index as module
    importlib.reload(module)

    result = module.get_item_details("DOC-NONEXISTENT")

    assert "error" in result
    assert "DOC-NONEXISTENT" in result["error"]
    assert "not found" in result["error"]
