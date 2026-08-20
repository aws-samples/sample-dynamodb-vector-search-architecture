"""CDK assertion tests for all stacks in the DynamoDB vector search architecture."""

import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import pytest

from stacks.dynamodb_stack import DynamoDBStack
from stacks.lambda_stack import LambdaStack
from stacks.bedrock_agent_stack import BedrockAgentStack


@pytest.fixture(scope="module")
def templates():
    """Synthesize all stacks with the same wiring as app.py and return templates."""
    app = cdk.App()

    # Set context values matching cdk.json
    app.node.set_context("project_prefix", "unified-agent")
    app.node.set_context("environment", "dev")
    app.node.set_context("agent_foundation_model", "anthropic.claude-3-haiku-20240307-v1:0")
    app.node.set_context("embedding_model_id", "amazon.titan-embed-text-v2:0")
    app.node.set_context("embedding_dimensions", "1024")
    app.node.set_context("vector_distance_function", "COSINE")
    app.node.set_context("vector_index_name", "content-embedding-index")
    app.node.set_context("table_name_suffix", "unified-agent-data")

    prefix = "unified-agent"

    # DynamoDB stack
    dynamodb_stack = DynamoDBStack(app, f"{prefix}-dynamodb")

    # Lambda stack
    lambda_stack = LambdaStack(
        app,
        f"{prefix}-lambda",
        table_name=dynamodb_stack.table_name,
        table_arn=dynamodb_stack.table_arn,
        table_stream_arn=dynamodb_stack.table_stream_arn,
        index_name=dynamodb_stack.index_name,
    )

    # Bedrock agent stack
    bedrock_agent_stack = BedrockAgentStack(
        app,
        f"{prefix}-bedrock-agent",
        action_group_lambda_arn=lambda_stack.action_group_lambda_arn,
    )

    dynamodb_template = Template.from_stack(dynamodb_stack)
    lambda_template = Template.from_stack(lambda_stack)
    bedrock_template = Template.from_stack(bedrock_agent_stack)

    return {
        "dynamodb": dynamodb_template,
        "lambda": lambda_template,
        "bedrock": bedrock_template,
    }


def test_dynamodb_table_has_pay_per_request_billing(templates):
    """Assert the DynamoDB table uses PAY_PER_REQUEST billing mode."""
    templates["dynamodb"].has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
        },
    )


def test_dynamodb_table_has_stream_enabled(templates):
    """Assert the DynamoDB table has streams enabled with NEW_AND_OLD_IMAGES."""
    templates["dynamodb"].has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "StreamSpecification": {
                "StreamViewType": "NEW_AND_OLD_IMAGES",
            },
        },
    )


def test_embedding_lambda_has_correct_environment_variables(templates):
    """Assert the embedding pipeline Lambda has required environment variables."""
    templates["lambda"].has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "index.handler",
            "Timeout": 60,
            "Environment": {
                "Variables": Match.object_like(
                    {
                        "TABLE_NAME": Match.any_value(),
                        "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                        "EMBEDDING_DIMENSIONS": "1024",
                    }
                ),
            },
        },
    )


def test_action_group_lambda_has_bedrock_permission(templates):
    """Assert a resource-based policy allows bedrock.amazonaws.com to invoke the action group Lambda."""
    templates["lambda"].has_resource_properties(
        "AWS::Lambda::Permission",
        {
            "Action": "lambda:InvokeFunction",
            "Principal": "bedrock.amazonaws.com",
        },
    )


def test_iam_policies_no_wildcard_dynamodb(templates):
    """Assert that IAM policies with DynamoDB actions do not use '*' as the resource.

    Note: dynamodb:ListStreams is excluded because AWS does not support
    resource-level permissions for this action, so CDK correctly uses '*'.
    """
    # Actions that AWS documents as not supporting resource-level permissions
    list_only_actions = {"dynamodb:ListStreams"}

    lambda_template = templates["lambda"]
    policies = lambda_template.find_resources("AWS::IAM::Policy")

    for policy_id, policy_resource in policies.items():
        statements = (
            policy_resource.get("Properties", {})
            .get("PolicyDocument", {})
            .get("Statement", [])
        )
        for statement in statements:
            actions = statement.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]

            # Filter to DynamoDB actions that support resource-level permissions
            scoped_dynamodb_actions = [
                a for a in actions
                if a.startswith("dynamodb:") and a not in list_only_actions
            ]

            if scoped_dynamodb_actions:
                resources = statement.get("Resource", [])
                if isinstance(resources, str):
                    resources = [resources]
                for resource in resources:
                    if isinstance(resource, str):
                        assert resource != "*", (
                            f"Policy '{policy_id}' uses wildcard '*' resource "
                            f"for DynamoDB actions: {scoped_dynamodb_actions}"
                        )


def test_lambda_timeout_values(templates):
    """Assert embedding Lambda timeout is 60s and action group Lambda timeout is 30s."""
    lambda_template = templates["lambda"]
    functions = lambda_template.find_resources("AWS::Lambda::Function")

    timeouts_found = set()
    for fn_id, fn_resource in functions.items():
        timeout = fn_resource.get("Properties", {}).get("Timeout")
        if timeout is not None:
            timeouts_found.add(timeout)

    assert 60 in timeouts_found, "Embedding pipeline Lambda should have 60s timeout"
    assert 30 in timeouts_found, "Action group Lambda should have 30s timeout"


def test_bedrock_agent_resource_created(templates):
    """Assert that a Bedrock Agent resource exists with the correct foundation model."""
    templates["bedrock"].has_resource_properties(
        "AWS::Bedrock::Agent",
        {
            "FoundationModel": "anthropic.claude-3-haiku-20240307-v1:0",
        },
    )
