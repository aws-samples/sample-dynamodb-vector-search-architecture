"""Action group Lambda function for Bedrock agent.

Handles semantic search and document retrieval operations invoked by the
Amazon Bedrock agent action group. Routes function calls to the appropriate
handler and returns results in the Bedrock agent response format.
"""

import json
import logging
import os

import boto3
from boto3.dynamodb.types import TypeDeserializer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ["TABLE_NAME"]
INDEX_NAME = os.environ["INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
EMBEDDING_DIMENSIONS = int(os.environ["EMBEDDING_DIMENSIONS"])

dynamodb = boto3.client("dynamodb")
bedrock_runtime = boto3.client("bedrock-runtime")

deserializer = TypeDeserializer()


def deserialize_dynamodb_item(dynamodb_item: dict) -> dict:
    """Convert a DynamoDB JSON item to a standard Python dictionary."""
    return {key: deserializer.deserialize(value) for key, value in dynamodb_item.items()}


def generate_embedding(text: str) -> list:
    """Generate a vector embedding for the given text using Bedrock.

    Args:
        text: The text content to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    request_body = json.dumps({
        "inputText": text,
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalize": True,
    })

    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=request_body,
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


def semantic_search(query: str, category: str, max_results: int = 5) -> list:
    """Search documents by semantic similarity using DynamoDB vector search.

    Args:
        query: Natural language search query.
        category: Document category to filter results.
        max_results: Maximum number of results to return (capped at 100).

    Returns:
        A list of matching documents with entity_id, title, and relevance score.
    """
    logger.info(
        "Semantic search: query='%s', category='%s', max_results=%d",
        query,
        category,
        max_results,
    )

    embedding = generate_embedding(query)
    top_k = min(max_results, 100)

    # SearchVector requires DynamoDB Number list format
    search_vector = [{"N": str(v)} for v in embedding]

    response = dynamodb.search_vectors(
        TableName=TABLE_NAME,
        IndexName=INDEX_NAME,
        SearchVector=search_vector,
        TopK=top_k,
        SearchConditionExpression="category = :cat",
        ExpressionAttributeValues={":cat": {"S": category}},
        ProjectionExpression="entity_id, title, category, entity_type",
    )

    results = []
    for search_result in response.get("SearchResults", []):
        item = search_result.get("Item", {})
        deserialized = deserialize_dynamodb_item(item)
        results.append({
            "entity_id": deserialized.get("entity_id"),
            "title": deserialized.get("title"),
            "score": search_result.get("Score", 0),
        })

    logger.info("Semantic search returned %d results", len(results))
    return results


def get_item_details(entity_id: str) -> dict:
    """Retrieve a document by its unique identifier.

    Args:
        entity_id: The unique identifier of the document.

    Returns:
        The document attributes (excluding embedding) or an error dict.
    """
    logger.info("Getting item details for entity_id=%s", entity_id)

    response = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={
            "entity_id": {"S": entity_id},
            "sk": {"S": "METADATA"},
        },
    )

    if "Item" not in response:
        logger.info("Item not found: entity_id=%s", entity_id)
        return {"error": f"Document {entity_id} not found"}

    item = deserialize_dynamodb_item(response["Item"])

    # Remove embedding from response as it is large and not useful for display
    item.pop("embedding", None)

    logger.info("Retrieved item: entity_id=%s", entity_id)
    return item


def format_agent_response(action_group: str, function_name: str, result: dict) -> dict:
    """Format the response in Bedrock agent action group format.

    Args:
        action_group: The name of the action group.
        function_name: The name of the function that was called.
        result: The result data to include in the response.

    Returns:
        A formatted response dictionary for the Bedrock agent.
    """
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "function": function_name,
            "functionResponse": {
                "responseBody": {
                    "TEXT": {
                        "body": json.dumps(result),
                    }
                }
            },
        },
    }


def handler(event, context):
    """Handle Bedrock agent action group invocations.

    Routes function calls to the appropriate handler based on the function
    name in the event, and returns results in the Bedrock agent response format.
    """
    logger.info("Received event: %s", json.dumps(event))

    action_group = event.get("actionGroup", "")
    function_name = event.get("function", "")

    # Extract parameters from the event parameters list
    parameters = {}
    for param in event.get("parameters", []):
        parameters[param["name"]] = param["value"]

    logger.info(
        "Processing function: %s, action_group: %s, parameters: %s",
        function_name,
        action_group,
        json.dumps(parameters),
    )

    if function_name == "semantic_search":
        query = parameters.get("query", "")
        category = parameters.get("category", "")
        result = semantic_search(query, category)
    elif function_name == "get_item_details":
        entity_id = parameters.get("entity_id", "")
        result = get_item_details(entity_id)
    else:
        logger.error("Unknown function: %s", function_name)
        result = {"error": f"Unknown function: {function_name}"}

    return format_agent_response(action_group, function_name, result)
