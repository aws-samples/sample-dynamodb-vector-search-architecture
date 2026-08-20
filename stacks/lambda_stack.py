import os

from aws_cdk import (
    Stack,
    Duration,
    BundlingOptions,
    ILocalBundling,
    aws_lambda as _lambda,
    aws_lambda_event_sources as lambda_event_sources,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
)
from constructs import Construct
import jsii
import subprocess


@jsii.implements(ILocalBundling)
class _LocalBundling:
    """Local bundling fallback when Docker is unavailable."""

    def try_bundle(self, output_dir: str, *, image, command=None, entrypoint=None,
                   environment=None, local=None, output_type=None,
                   security_opt=None, user=None, volumes=None,
                   volumes_from=None, working_directory=None,
                   bundling_file_access=None, network=None) -> bool:
        python_dir = os.path.join(output_dir, "python")
        os.makedirs(python_dir, exist_ok=True)
        subprocess.check_call(
            ["pip", "install", "-r", "requirements.txt", "-t", python_dir, "--quiet"],
            cwd=os.path.join(os.path.dirname(os.path.dirname(__file__)), "layers", "boto3"),
        )
        return True


class LambdaStack(Stack):
    """Lambda functions for embedding pipeline and action group."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        table_name: str,
        table_arn: str,
        table_stream_arn: str,
        index_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._table_name = table_name
        self._table_arn = table_arn
        self._table_stream_arn = table_stream_arn
        self._index_name = index_name

        # Read context values
        embedding_model_id = (
            self.node.try_get_context("embedding_model_id")
            or "amazon.titan-embed-text-v2:0"
        )
        embedding_dimensions = (
            self.node.try_get_context("embedding_dimensions") or "1024"
        )

        # Path to lambda_functions and layers directories
        lambda_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "lambda_functions"
        )
        layers_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "layers"
        )

        # Shared boto3 layer with latest SDK including DynamoDB vector search support
        boto3_layer = _lambda.LayerVersion(
            self,
            "Boto3Layer",
            code=_lambda.Code.from_asset(
                os.path.join(layers_dir, "boto3"),
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install -r requirements.txt -t /asset-output/python"
                        " && cp -au . /asset-output/python/",
                    ],
                    local=_LocalBundling(),
                ),
            ),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="Boto3 layer with latest SDK including DynamoDB SearchVectors support",
        )

        # --- Embedding Pipeline Lambda ---
        self._embedding_pipeline_fn = _lambda.Function(
            self,
            "EmbeddingPipelineFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=_lambda.Code.from_asset(
                os.path.join(lambda_dir, "embedding_pipeline")
            ),
            layers=[boto3_layer],
            memory_size=512,
            timeout=Duration.seconds(60),
            environment={
                "TABLE_NAME": table_name,
                "EMBEDDING_MODEL_ID": embedding_model_id,
                "EMBEDDING_DIMENSIONS": str(embedding_dimensions),
            },
        )

        # DynamoDB Streams event source for embedding pipeline
        # Import table reference from ARN for the event source
        imported_table = dynamodb.Table.from_table_attributes(
            self,
            "ImportedTable",
            table_name=table_name,
            table_stream_arn=table_stream_arn,
        )

        self._embedding_pipeline_fn.add_event_source(
            lambda_event_sources.DynamoEventSource(
                imported_table,
                starting_position=_lambda.StartingPosition.TRIM_HORIZON,
                batch_size=10,
                max_batching_window=Duration.seconds(30),
                retry_attempts=3,
                bisect_batch_on_error=True,
                filters=[
                    _lambda.FilterCriteria.filter(
                        {
                            "eventName": _lambda.FilterRule.or_("INSERT", "MODIFY"),
                        }
                    )
                ],
            )
        )

        # IAM permissions for embedding pipeline Lambda
        self._embedding_pipeline_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:UpdateItem"],
                resources=[table_arn],
            )
        )

        self._embedding_pipeline_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:DescribeStream",
                    "dynamodb:GetRecords",
                    "dynamodb:GetShardIterator",
                    "dynamodb:ListStreams",
                ],
                resources=[f"{table_arn}/stream/*"],
            )
        )

        self._embedding_pipeline_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            )
        )

        # --- Action Group Lambda ---
        self._action_group_fn = _lambda.Function(
            self,
            "ActionGroupFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=_lambda.Code.from_asset(
                os.path.join(lambda_dir, "action_group")
            ),
            layers=[boto3_layer],
            memory_size=512,
            timeout=Duration.seconds(30),
            environment={
                "TABLE_NAME": table_name,
                "INDEX_NAME": index_name,
                "EMBEDDING_MODEL_ID": embedding_model_id,
                "EMBEDDING_DIMENSIONS": str(embedding_dimensions),
            },
        )

        # IAM permissions for action group Lambda
        self._action_group_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                ],
                resources=[table_arn],
            )
        )

        self._action_group_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:SearchVectors"],
                resources=[f"{table_arn}/index/{index_name}"],
            )
        )

        self._action_group_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            )
        )

        # Resource-based policy allowing Bedrock to invoke the action group Lambda
        self._action_group_fn.add_permission(
            "BedrockInvokePermission",
            principal=iam.ServicePrincipal("bedrock.amazonaws.com"),
            action="lambda:InvokeFunction",
        )

    @property
    def action_group_lambda_arn(self) -> str:
        """The ARN of the action group Lambda function."""
        return self._action_group_fn.function_arn
