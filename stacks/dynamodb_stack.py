from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_dynamodb as dynamodb,
    custom_resources as cr,
    aws_iam as iam,
)
from constructs import Construct


class DynamoDBStack(Stack):
    """DynamoDB table with native vector index for semantic search."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Read context values
        prefix = self.node.try_get_context("project_prefix") or "unified-agent"
        table_name_suffix = (
            self.node.try_get_context("table_name_suffix") or "unified-agent-data"
        )
        self._index_name = (
            self.node.try_get_context("vector_index_name")
            or "content-embedding-index"
        )
        embedding_dimensions = int(
            self.node.try_get_context("embedding_dimensions") or "1024"
        )
        distance_function = (
            self.node.try_get_context("vector_distance_function") or "COSINE"
        )

        # Create the DynamoDB table
        self._table = dynamodb.Table(
            self,
            "VectorSearchTable",
            table_name=f"{prefix}-{table_name_suffix}",
            partition_key=dynamodb.Attribute(
                name="entity_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Create custom resource to add the vector index via UpdateTable API.
        # AttributeDefinitions for SearchSchema attributes (category,
        # entity_type) are passed in the UpdateTable call rather than at
        # table creation time, because CloudFormation's CreateTable only
        # allows attributes referenced in the base table KeySchema.
        vector_index_cr = cr.AwsCustomResource(
            self,
            "VectorIndexCustomResource",
            on_create=cr.AwsSdkCall(
                service="DynamoDB",
                action="updateTable",
                parameters={
                    "TableName": self._table.table_name,
                    "AttributeDefinitions": [
                        {"AttributeName": "category", "AttributeType": "S"},
                        {
                            "AttributeName": "entity_type",
                            "AttributeType": "S",
                        },
                    ],
                    "VectorIndexUpdates": [
                        {
                            "Create": {
                                "IndexName": self._index_name,
                                "VectorAttribute": {
                                    "AttributeName": "embedding"
                                },
                                "Dimensions": embedding_dimensions,
                                "DistanceFunction": distance_function,
                                "SearchSchema": [
                                    {
                                        "AttributeName": "category",
                                        "SearchSchemaElementType": "HASH",
                                    },
                                    {
                                        "AttributeName": "entity_type",
                                        "SearchSchemaElementType": "INLINE_FILTER",
                                    },
                                ],
                                "Projection": {"ProjectionType": "ALL"},
                            }
                        }
                    ],
                },
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"{self._table.table_name}-vector-index"
                ),
            ),
            on_delete=cr.AwsSdkCall(
                service="DynamoDB",
                action="updateTable",
                parameters={
                    "TableName": self._table.table_name,
                    "VectorIndexUpdates": [
                        {"Delete": {"IndexName": self._index_name}}
                    ],
                },
                ignore_error_codes_matching="ResourceNotFoundException",
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["dynamodb:UpdateTable"],
                        resources=[self._table.table_arn],
                    )
                ]
            ),
        )

        # Ensure the custom resource runs after the table is created
        vector_index_cr.node.add_dependency(self._table)

        # CloudFormation outputs
        CfnOutput(self, "TableName", value=self._table.table_name)
        CfnOutput(self, "TableArn", value=self._table.table_arn)
        CfnOutput(
            self, "TableStreamArn", value=self._table.table_stream_arn
        )
        CfnOutput(self, "IndexName", value=self._index_name)

    @property
    def table_name(self) -> str:
        """The name of the DynamoDB table."""
        return self._table.table_name

    @property
    def table_arn(self) -> str:
        """The ARN of the DynamoDB table."""
        return self._table.table_arn

    @property
    def table_stream_arn(self) -> str:
        """The ARN of the DynamoDB table stream."""
        return self._table.table_stream_arn

    @property
    def index_name(self) -> str:
        """The name of the vector index."""
        return self._index_name
