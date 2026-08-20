#!/usr/bin/env python3
import aws_cdk as cdk

from stacks.dynamodb_stack import DynamoDBStack
from stacks.lambda_stack import LambdaStack
from stacks.bedrock_agent_stack import BedrockAgentStack

app = cdk.App()

prefix = app.node.try_get_context("project_prefix") or "unified-agent"
environment = app.node.try_get_context("environment") or "dev"

# DynamoDB stack (foundation layer)
dynamodb_stack = DynamoDBStack(
    app,
    f"{prefix}-dynamodb",
)

# Lambda stack (depends on DynamoDB stack)
lambda_stack = LambdaStack(
    app,
    f"{prefix}-lambda",
    table_name=dynamodb_stack.table_name,
    table_arn=dynamodb_stack.table_arn,
    table_stream_arn=dynamodb_stack.table_stream_arn,
    index_name=dynamodb_stack.index_name,
)
lambda_stack.add_stack_dependency(dynamodb_stack)

# Bedrock agent stack (depends on Lambda stack)
bedrock_agent_stack = BedrockAgentStack(
    app,
    f"{prefix}-bedrock-agent",
    action_group_lambda_arn=lambda_stack.action_group_lambda_arn,
)
bedrock_agent_stack.add_stack_dependency(lambda_stack)

# Apply tags to all stacks
for stack in [dynamodb_stack, lambda_stack, bedrock_agent_stack]:
    cdk.Tags.of(stack).add("Project", "dynamodb-vector-search-agent")
    cdk.Tags.of(stack).add("Environment", environment)

app.synth()
