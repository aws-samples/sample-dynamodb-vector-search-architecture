"""Embedding pipeline Lambda function.

Triggered by DynamoDB Streams, generates embeddings for new or modified items
using Amazon Titan Text Embeddings V2 and writes them back to the table.
"""

import json
import logging
import os

import boto3
from boto3.dynamodb.types import TypeDeserializer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ["TABLE_NAME"]
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


def handler(event, context):
    """Process DynamoDB Streams records and generate embeddings.

    For each INSERT or MODIFY record with a content field, generates an
    embedding vector and writes it back to the item. Includes an infinite-loop
    guard to avoid reprocessing items where only the embedding changed.
    """
    records = event.get("Records", [])
    logger.info("Processing %d records", len(records))

    for record in records:
        entity_id = "unknown"
        try:
            event_name = record.get("eventName")
            if event_name not in ("INSERT", "MODIFY"):
                logger.info("Skipping event type: %s", event_name)
                continue

            new_image_raw = record["dynamodb"].get("NewImage", {})
            old_image_raw = record["dynamodb"].get("OldImage", {})

            new_image = deserialize_dynamodb_item(new_image_raw)
            old_image = deserialize_dynamodb_item(old_image_raw)

            entity_id = new_image.get("entity_id", "unknown")
            sk = new_image.get("sk", "unknown")

            logger.info(
                "Processing record: entity_id=%s, sk=%s, event=%s",
                entity_id,
                sk,
                event_name,
            )

            # Infinite-loop guard: skip if embedding already exists and content
            # is unchanged. This prevents reprocessing when the Lambda writes
            # the embedding back and triggers another MODIFY event.
            if "embedding" in new_image and old_image.get("content") == new_image.get("content"):
                logger.info(
                    "Skipping entity_id=%s - content unchanged, embedding already exists",
                    entity_id,
                )
                continue

            content = new_image.get("content")
            if not content:
                logger.info("Skipping entity_id=%s - no content field", entity_id)
                continue

            logger.info("Generating embedding for entity_id=%s", entity_id)
            embedding = generate_embedding(content)

            # Write embedding back as a DynamoDB List of Numbers
            dynamodb.update_item(
                TableName=TABLE_NAME,
                Key={
                    "entity_id": {"S": str(entity_id)},
                    "sk": {"S": str(sk)},
                },
                UpdateExpression="SET embedding = :emb",
                ExpressionAttributeValues={
                    ":emb": {"L": [{"N": str(v)} for v in embedding]},
                },
            )

            logger.info(
                "Embedding written for entity_id=%s (%d dimensions)",
                entity_id,
                len(embedding),
            )

        except Exception as e:
            logger.error(
                "Error processing record entity_id=%s: %s",
                entity_id,
                str(e),
                exc_info=True,
            )
            continue

    logger.info("Finished processing batch")
