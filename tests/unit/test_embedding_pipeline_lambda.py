"""Unit tests for the embedding pipeline Lambda function."""

import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing the module
os.environ["TABLE_NAME"] = "test-table"
os.environ["EMBEDDING_MODEL_ID"] = "amazon.titan-embed-text-v2:0"
os.environ["EMBEDDING_DIMENSIONS"] = "1024"


def make_streaming_body(data: dict) -> MagicMock:
    """Create a mock StreamingBody that returns JSON data on read()."""
    body = MagicMock()
    body.read.return_value = json.dumps(data).encode("utf-8")
    return body


def make_fake_embedding(dimensions: int = 1024) -> list:
    """Generate a fake embedding vector."""
    return [0.1 * (i % 10) for i in range(dimensions)]


def build_streams_record(
    event_name: str,
    new_image: dict,
    old_image: dict | None = None,
) -> dict:
    """Build a DynamoDB Streams record in the expected format.

    Args:
        event_name: INSERT, MODIFY, or REMOVE.
        new_image: DynamoDB JSON format for NewImage.
        old_image: DynamoDB JSON format for OldImage.
    """
    record = {
        "eventName": event_name,
        "dynamodb": {
            "NewImage": new_image,
        },
    }
    if old_image is not None:
        record["dynamodb"]["OldImage"] = old_image
    else:
        record["dynamodb"]["OldImage"] = {}
    return record


@patch("boto3.client")
def test_insert_event_triggers_embedding_generation(mock_boto_client):
    """An INSERT event with a content field generates and writes an embedding."""
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

    # Re-import to pick up patched boto3
    import importlib
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            build_streams_record(
                event_name="INSERT",
                new_image={
                    "entity_id": {"S": "DOC-001"},
                    "sk": {"S": "METADATA"},
                    "title": {"S": "Test Document"},
                    "content": {"S": "This is test content for embedding generation."},
                    "category": {"S": "runbook"},
                },
            )
        ]
    }

    module.handler(event, None)

    mock_bedrock.invoke_model.assert_called_once()
    mock_dynamodb.update_item.assert_called_once()

    call_kwargs = mock_dynamodb.update_item.call_args[1]
    assert call_kwargs["TableName"] == "test-table"
    assert call_kwargs["Key"] == {
        "entity_id": {"S": "DOC-001"},
        "sk": {"S": "METADATA"},
    }
    assert call_kwargs["UpdateExpression"] == "SET embedding = :emb"


@patch("boto3.client")
def test_modify_event_with_changed_content_triggers_embedding(mock_boto_client):
    """A MODIFY event with changed content generates a new embedding."""
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

    import importlib
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            build_streams_record(
                event_name="MODIFY",
                new_image={
                    "entity_id": {"S": "DOC-002"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": "Updated content that differs from old."},
                    "category": {"S": "architecture"},
                },
                old_image={
                    "entity_id": {"S": "DOC-002"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": "Original content before modification."},
                    "category": {"S": "architecture"},
                },
            )
        ]
    }

    module.handler(event, None)

    mock_bedrock.invoke_model.assert_called_once()
    mock_dynamodb.update_item.assert_called_once()


@patch("boto3.client")
def test_modify_event_embedding_exists_content_unchanged_is_skipped(mock_boto_client):
    """Infinite-loop guard: skip when embedding exists and content is unchanged."""
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
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    same_content = "This content has not changed."
    event = {
        "Records": [
            build_streams_record(
                event_name="MODIFY",
                new_image={
                    "entity_id": {"S": "DOC-003"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": same_content},
                    "embedding": {"L": [{"N": "0.1"}, {"N": "0.2"}]},
                },
                old_image={
                    "entity_id": {"S": "DOC-003"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": same_content},
                },
            )
        ]
    }

    module.handler(event, None)

    mock_dynamodb.update_item.assert_not_called()
    mock_bedrock.invoke_model.assert_not_called()


@patch("boto3.client")
def test_record_without_content_field_is_skipped(mock_boto_client):
    """A record missing the content field is skipped without error."""
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
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            build_streams_record(
                event_name="INSERT",
                new_image={
                    "entity_id": {"S": "DOC-004"},
                    "sk": {"S": "METADATA"},
                    "title": {"S": "No content here"},
                    "category": {"S": "runbook"},
                },
            )
        ]
    }

    module.handler(event, None)

    mock_dynamodb.update_item.assert_not_called()
    mock_bedrock.invoke_model.assert_not_called()


@patch("boto3.client")
def test_delete_event_is_ignored(mock_boto_client):
    """A REMOVE event is ignored entirely."""
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
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            {
                "eventName": "REMOVE",
                "dynamodb": {
                    "OldImage": {
                        "entity_id": {"S": "DOC-005"},
                        "sk": {"S": "METADATA"},
                        "content": {"S": "Some content"},
                    },
                },
            }
        ]
    }

    module.handler(event, None)

    mock_dynamodb.update_item.assert_not_called()
    mock_bedrock.invoke_model.assert_not_called()


@patch("boto3.client")
def test_update_item_called_with_correct_format(mock_boto_client):
    """Verify embedding is stored in DynamoDB List of Numbers format."""
    mock_dynamodb = MagicMock()
    mock_bedrock = MagicMock()

    def client_factory(service_name, **kwargs):
        if service_name == "dynamodb":
            return mock_dynamodb
        elif service_name == "bedrock-runtime":
            return mock_bedrock
        return MagicMock()

    mock_boto_client.side_effect = client_factory

    fake_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_bedrock.invoke_model.return_value = {
        "body": make_streaming_body({"embedding": fake_embedding})
    }

    import importlib
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            build_streams_record(
                event_name="INSERT",
                new_image={
                    "entity_id": {"S": "DOC-006"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": "Content for format verification."},
                },
            )
        ]
    }

    module.handler(event, None)

    call_kwargs = mock_dynamodb.update_item.call_args[1]
    emb_value = call_kwargs["ExpressionAttributeValues"][":emb"]

    # Verify the embedding is stored as {"L": [{"N": "..."}, ...]}
    assert "L" in emb_value
    assert len(emb_value["L"]) == 5
    for item in emb_value["L"]:
        assert "N" in item
        # Verify values are string representations of floats
        float(item["N"])


@patch("boto3.client")
def test_error_in_one_record_does_not_block_subsequent(mock_boto_client):
    """An error processing one record does not prevent the next record."""
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

    # First call raises, second call succeeds
    mock_bedrock.invoke_model.side_effect = [
        Exception("Bedrock service error"),
        {"body": make_streaming_body({"embedding": fake_embedding})},
    ]

    import importlib
    import lambda_functions.embedding_pipeline.index as module
    importlib.reload(module)

    event = {
        "Records": [
            build_streams_record(
                event_name="INSERT",
                new_image={
                    "entity_id": {"S": "DOC-007"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": "First record that will fail."},
                },
            ),
            build_streams_record(
                event_name="INSERT",
                new_image={
                    "entity_id": {"S": "DOC-008"},
                    "sk": {"S": "METADATA"},
                    "content": {"S": "Second record that should succeed."},
                },
            ),
        ]
    }

    # Should not raise an exception
    module.handler(event, None)

    # Second record should still be processed
    assert mock_bedrock.invoke_model.call_count == 2
    mock_dynamodb.update_item.assert_called_once()
    call_kwargs = mock_dynamodb.update_item.call_args[1]
    assert call_kwargs["Key"]["entity_id"]["S"] == "DOC-008"
