import json
import os

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_bedrock as bedrock,
    aws_iam as iam,
)
from constructs import Construct


class BedrockAgentStack(Stack):
    """Bedrock agent with action group for semantic search operations."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        action_group_lambda_arn: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._action_group_lambda_arn = action_group_lambda_arn

        # Read context values
        project_prefix = self.node.try_get_context("project_prefix") or "unified-agent"
        agent_foundation_model = (
            self.node.try_get_context("agent_foundation_model")
            or "anthropic.claude-3-haiku-20240307-v1:0"
        )

        # --- Agent IAM role ---
        agent_role = iam.Role(
            self,
            "AgentRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            description="IAM role for the Bedrock knowledge agent",
        )

        agent_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:*::foundation-model/{agent_foundation_model}"
                ],
            )
        )

        # --- Load action group schema ---
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "schemas",
            "action_group_schema.json",
        )
        with open(schema_path, "r") as f:
            action_group_schema = json.load(f)

        # Build FunctionProperty objects from the schema
        function_properties = []
        for func in action_group_schema["functions"]:
            parameters = {}
            for param_name, param_def in func.get("parameters", {}).items():
                parameters[param_name] = (
                    bedrock.CfnAgent.ParameterDetailProperty(
                        type=param_def["type"],
                        description=param_def.get("description", ""),
                        required=param_def.get("required", False),
                    )
                )

            function_properties.append(
                bedrock.CfnAgent.FunctionProperty(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=parameters,
                )
            )

        # --- Bedrock agent ---
        agent = bedrock.CfnAgent(
            self,
            "KnowledgeAgent",
            agent_name=f"{project_prefix}-knowledge-agent",
            foundation_model=agent_foundation_model,
            instruction=(
                "You are a knowledge assistant that helps users find and retrieve "
                "technical documents. You can search documents semantically by topic "
                "and category, or retrieve specific documents by their identifier. "
                "When searching, always ask for or infer the relevant category to "
                "narrow results. Present search results clearly with titles and "
                "relevance scores. When retrieving a specific document, present its "
                "full content in a readable format."
            ),
            idle_session_ttl_in_seconds=600,
            agent_resource_role_arn=agent_role.role_arn,
            action_groups=[
                bedrock.CfnAgent.AgentActionGroupProperty(
                    action_group_name="knowledge-operations",
                    action_group_executor=bedrock.CfnAgent.ActionGroupExecutorProperty(
                        lambda_=action_group_lambda_arn,
                    ),
                    function_schema=bedrock.CfnAgent.FunctionSchemaProperty(
                        functions=function_properties,
                    ),
                )
            ],
        )

        # --- Agent alias ---
        agent_alias = bedrock.CfnAgentAlias(
            self,
            "AgentAlias",
            agent_alias_name="live",
            agent_id=agent.attr_agent_id,
        )

        # --- Outputs ---
        CfnOutput(
            self,
            "AgentId",
            value=agent.attr_agent_id,
            description="Bedrock Agent ID",
        )
        CfnOutput(
            self,
            "AgentAliasId",
            value=agent_alias.attr_agent_alias_id,
            description="Bedrock Agent Alias ID",
        )
