#!/usr/bin/env python3
"""Seed sample documents into the DynamoDB table for testing vector search.

Usage:
    python scripts/seed_data.py --table-name unified-agent-unified-agent-data --region us-east-1 --wait
"""

import argparse
import json
import time

import boto3


SAMPLE_DOCUMENTS = [
    # Runbook category (5 documents)
    {
        "entity_id": {"S": "DOC-001"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Retry strategy for payment service failures"},
        "content": {"S": (
            "When the payment service returns HTTP 5xx errors or times out, follow this "
            "retry strategy to minimize customer impact. First, verify the failure type by "
            "checking the error code in the service response. For transient failures such as "
            "503 Service Unavailable or 504 Gateway Timeout, implement exponential backoff "
            "starting at 100 milliseconds with a maximum of three retries. Use jitter to "
            "prevent thundering herd scenarios across multiple callers. For idempotent "
            "operations like charge captures, include the idempotency key in retry attempts "
            "to prevent duplicate charges. If all retries fail, route the transaction to a "
            "dead-letter queue for manual review. Set CloudWatch alarms to trigger when the "
            "DLQ depth exceeds five messages within a ten-minute window. Escalate to the "
            "payments on-call engineer if the alarm fires during peak traffic hours."
        )},
        "category": {"S": "runbook"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-15T10:30:00Z"},
    },
    {
        "entity_id": {"S": "DOC-002"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Database failover procedure"},
        "content": {"S": (
            "This runbook covers the steps for performing a planned or unplanned database "
            "failover from the primary Amazon RDS instance to the read replica in a "
            "different Availability Zone. Before initiating failover, confirm replication "
            "lag is under 100 milliseconds by querying the ReplicaLag CloudWatch metric. "
            "For planned failovers, enable maintenance mode on the application load balancer "
            "to drain active connections over a 30-second window. Promote the read replica "
            "using the RDS console or the promote-read-replica CLI command. Update the "
            "application connection string in AWS Secrets Manager and trigger a rolling "
            "restart of application instances. Verify connectivity by running a health "
            "check query against the new primary. Monitor error rates and latency for "
            "15 minutes post-failover. If issues persist, invoke the rollback procedure "
            "documented in section four of this runbook."
        )},
        "category": {"S": "runbook"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-16T09:00:00Z"},
    },
    {
        "entity_id": {"S": "DOC-003"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Memory leak investigation steps"},
        "content": {"S": (
            "Use this runbook when a service instance shows steadily increasing memory "
            "consumption without corresponding traffic growth. Start by identifying the "
            "affected instances using the CloudWatch MemoryUtilization metric grouped by "
            "instance ID. Connect to the instance using Session Manager and capture a heap "
            "dump with jmap for Java services or generate a core dump for Python services. "
            "Analyze the heap dump with Eclipse MAT or similar tools to identify the "
            "dominator tree and retained objects. Common root causes include unbounded "
            "caches without eviction policies, event listeners that are never deregistered, "
            "and connection pools that grow beyond configured limits. After identifying the "
            "leak source, verify the fix in a staging environment by running a sustained "
            "load test for at least one hour while monitoring memory. Deploy the fix using "
            "a canary deployment with memory thresholds as rollback criteria."
        )},
        "category": {"S": "runbook"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-17T14:15:00Z"},
    },
    {
        "entity_id": {"S": "DOC-004"},
        "sk": {"S": "METADATA"},
        "title": {"S": "API rate limit escalation process"},
        "content": {"S": (
            "When downstream API rate limits are exceeded, follow this escalation process "
            "to restore service health. First, confirm the rate limit breach by checking "
            "HTTP 429 response codes in the application logs. Identify the specific API "
            "endpoint and caller responsible using request tracing data from X-Ray. "
            "Implement client-side throttling using a token bucket algorithm to smooth "
            "request bursts. If the traffic is legitimate and sustained, file a service "
            "limit increase request through the AWS Support console with business "
            "justification. For immediate relief, enable request queuing with a maximum "
            "queue depth of 1,000 messages and a visibility timeout of 60 seconds. Monitor "
            "the queue depth and processing latency to verify the backlog clears within "
            "the SLA window. Document the incident in the post-mortem template and schedule "
            "a review to implement longer-term capacity planning improvements."
        )},
        "category": {"S": "runbook"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-18T11:45:00Z"},
    },
    {
        "entity_id": {"S": "DOC-005"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Cache invalidation runbook"},
        "content": {"S": (
            "Follow this procedure when stale data is served from the ElastiCache Redis "
            "cluster. First, determine the scope of stale data by comparing cache entries "
            "with the source of truth in DynamoDB. Identify whether the issue affects a "
            "single key pattern or the entire cache namespace. For targeted invalidation, "
            "use the Redis SCAN command with a pattern match to identify affected keys, "
            "then delete them in batches of 100 to avoid blocking the Redis event loop. "
            "For full cache invalidation, use the flush-specific-db approach rather than "
            "FLUSHALL to prevent impact on other applications sharing the cluster. After "
            "invalidation, monitor cache hit rates and origin database query load to "
            "confirm the cache is repopulating correctly. Set a CloudWatch alarm on the "
            "cache hit ratio metric with a threshold below 60 percent to detect future "
            "invalidation events early. Update TTL values if the stale data resulted from "
            "overly aggressive caching policies."
        )},
        "category": {"S": "runbook"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-19T08:20:00Z"},
    },
    # Architecture category (5 documents)
    {
        "entity_id": {"S": "DOC-006"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Microservices event-driven architecture"},
        "content": {"S": (
            "This architecture uses Amazon EventBridge as the central event bus to decouple "
            "microservices and enable asynchronous communication. Each service publishes "
            "domain events to a shared event bus using a standardized envelope format "
            "containing source, detail-type, and a JSON payload. Consumer services subscribe "
            "to specific event patterns using EventBridge rules that route matching events "
            "to SQS queues for reliable processing. This pattern eliminates point-to-point "
            "API calls between services and provides natural backpressure through queue "
            "depth monitoring. For event ordering, partition events by aggregate ID and "
            "include a sequence number in the payload. Dead-letter queues capture failed "
            "processing attempts after three retries. The architecture supports gradual "
            "service decomposition by allowing new consumers to subscribe to existing "
            "events without modifying producers."
        )},
        "category": {"S": "architecture"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-20T13:00:00Z"},
    },
    {
        "entity_id": {"S": "DOC-007"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Multi-region active-active design"},
        "content": {"S": (
            "The active-active multi-region architecture distributes traffic across two AWS "
            "Regions using Amazon Route 53 latency-based routing. Each Region runs an "
            "independent application stack with its own DynamoDB global table for data "
            "replication. Write operations use a conflict resolution strategy based on "
            "last-writer-wins with vector clocks for causal ordering. Static assets are "
            "served from CloudFront with origin groups that automatically failover between "
            "Regions. The health check configuration uses Route 53 health checks with a "
            "failover threshold of three consecutive failures over 30 seconds. Regional "
            "isolation is maintained through separate VPCs with no cross-region peering, "
            "ensuring that a failure in one Region does not cascade. Deployment uses a "
            "phased approach where changes roll out to the secondary Region first, followed "
            "by the primary after a 15-minute observation period."
        )},
        "category": {"S": "architecture"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-21T10:30:00Z"},
    },
    {
        "entity_id": {"S": "DOC-008"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Data lake ingestion pipeline"},
        "content": {"S": (
            "This architecture implements a scalable data lake ingestion pipeline using "
            "Amazon Kinesis Data Streams for real-time event capture and AWS Glue for "
            "batch ETL processing. Raw events land in an S3 raw zone partitioned by "
            "date and source system. A Glue crawler automatically discovers schema changes "
            "and updates the Data Catalog. Transformation jobs convert raw JSON into "
            "Parquet format with Snappy compression, reducing storage costs by up to "
            "80 percent compared to raw JSON. The processed data lands in a curated zone "
            "organized by business domain. Amazon Athena provides serverless SQL access "
            "for ad-hoc queries, while Amazon Redshift Spectrum handles complex analytical "
            "workloads. Data quality checks run after each batch using Great Expectations "
            "to validate schema conformance, null rates, and value distributions. Failed "
            "quality checks quarantine affected partitions and send notifications through "
            "Amazon SNS."
        )},
        "category": {"S": "architecture"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-22T15:45:00Z"},
    },
    {
        "entity_id": {"S": "DOC-009"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Real-time fraud detection system"},
        "content": {"S": (
            "The fraud detection architecture processes transaction events in real time "
            "using Amazon Kinesis Data Streams with a Kinesis Data Analytics application "
            "running Apache Flink. The Flink application applies sliding window aggregations "
            "to compute features such as transaction velocity, geographic dispersion, and "
            "amount deviation from historical baselines. These features feed into an Amazon "
            "SageMaker endpoint hosting a gradient boosted tree model trained on labeled "
            "fraud data. Transactions scoring above the 0.85 confidence threshold are "
            "flagged for review and routed to an SQS queue for human analyst processing. "
            "The system maintains sub-second latency from event ingestion to fraud score "
            "generation by using Flink checkpointing with a one-second interval. Model "
            "retraining runs weekly using the latest labeled data and deploys through a "
            "blue-green endpoint configuration to avoid inference downtime."
        )},
        "category": {"S": "architecture"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-23T09:15:00Z"},
    },
    {
        "entity_id": {"S": "DOC-010"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Serverless API gateway pattern"},
        "content": {"S": (
            "This architecture pattern uses Amazon API Gateway with AWS Lambda to build "
            "a scalable REST API without managing servers. API Gateway handles request "
            "routing, authentication via Amazon Cognito user pools, request validation, "
            "and rate limiting. Each API resource maps to a dedicated Lambda function "
            "following the single-responsibility principle. Lambda functions connect to "
            "DynamoDB for data persistence using the on-demand capacity mode to match "
            "unpredictable traffic patterns. Response caching at the API Gateway level "
            "reduces Lambda invocations for frequently requested resources. Custom "
            "authorizers implement fine-grained access control based on JWT claims. "
            "CloudWatch alarms monitor API latency at the p99 level and Lambda error "
            "rates. The deployment pipeline uses SAM with canary deployments that shift "
            "10 percent of traffic to new versions and automatically roll back if the "
            "error rate exceeds one percent over five minutes."
        )},
        "category": {"S": "architecture"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-24T11:00:00Z"},
    },
    # Troubleshooting category (5 documents)
    {
        "entity_id": {"S": "DOC-011"},
        "sk": {"S": "METADATA"},
        "title": {"S": "High latency in DynamoDB queries"},
        "content": {"S": (
            "Diagnose high latency in DynamoDB queries by first identifying whether the "
            "issue is at the table level or specific to certain access patterns. Check "
            "the SuccessfulRequestLatency metric in CloudWatch grouped by operation type "
            "to isolate Query versus Scan operations. For Query operations, verify that "
            "the partition key provides sufficient cardinality to distribute load evenly "
            "across partitions. Hot partitions manifest as throttling events visible in "
            "the ThrottledRequests metric. Review the query filter expressions to confirm "
            "filtering happens at the DynamoDB level rather than client-side. Large result "
            "sets cause pagination overhead; consider adding a more selective sort key "
            "condition to reduce items evaluated. If using Global Secondary Indexes, "
            "check for GSI backpressure by monitoring the GSI throttle metrics. Enable "
            "DynamoDB Contributor Insights to identify the most accessed partition keys "
            "and adjust the data model accordingly."
        )},
        "category": {"S": "troubleshooting"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-25T08:30:00Z"},
    },
    {
        "entity_id": {"S": "DOC-012"},
        "sk": {"S": "METADATA"},
        "title": {"S": "Lambda cold start optimization"},
        "content": {"S": (
            "Reduce Lambda cold start latency by analyzing the function initialization "
            "phase using CloudWatch Logs Insights. Query for INIT_START and REPORT log "
            "entries to measure init duration separately from execution duration. For "
            "Python runtimes, move heavyweight imports inside the handler function if "
            "they are conditionally used. Use lazy initialization for SDK clients by "
            "creating them outside the handler but deferring actual connection until "
            "first use. Reduce deployment package size by excluding development "
            "dependencies and using Lambda layers for shared libraries. Enable "
            "Provisioned Concurrency for latency-sensitive functions with predictable "
            "traffic patterns, setting the concurrency level to match the p95 concurrent "
            "execution count. For burst traffic scenarios, use Application Auto Scaling "
            "with Provisioned Concurrency to scale based on utilization. Consider "
            "switching to ARM64 architecture for up to 20 percent faster initialization "
            "at lower cost."
        )},
        "category": {"S": "troubleshooting"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-26T14:00:00Z"},
    },
    {
        "entity_id": {"S": "DOC-013"},
        "sk": {"S": "METADATA"},
        "title": {"S": "S3 access denied errors"},
        "content": {"S": (
            "Troubleshoot S3 AccessDenied errors by systematically checking each layer "
            "of the access control chain. Start with the IAM policy attached to the "
            "principal making the request. Use the IAM Policy Simulator to test whether "
            "the policy allows the specific S3 action on the target resource ARN. Next, "
            "check the S3 bucket policy for explicit deny statements that might override "
            "IAM allows. Verify that the bucket does not have Block Public Access settings "
            "that conflict with the intended access pattern. For cross-account access, "
            "confirm both the source account IAM policy and the destination bucket policy "
            "grant the necessary permissions. Check for S3 Object Ownership settings that "
            "might affect ACL-based access. If using VPC endpoints, verify the endpoint "
            "policy allows the action. Enable CloudTrail S3 data events to capture the "
            "exact API call and error details for forensic analysis."
        )},
        "category": {"S": "troubleshooting"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-27T10:45:00Z"},
    },
    {
        "entity_id": {"S": "DOC-014"},
        "sk": {"S": "METADATA"},
        "title": {"S": "ECS task placement failures"},
        "content": {"S": (
            "Investigate ECS task placement failures by examining the stopped task reasons "
            "in the ECS console or through the describe-tasks CLI command. Common failure "
            "reasons include insufficient CPU or memory resources on container instances, "
            "port conflicts when using host networking mode, and placement constraint "
            "violations. For capacity issues, check the cluster's registered versus "
            "available resources using describe-container-instances. If using Fargate, "
            "verify the task definition specifies a valid CPU and memory combination from "
            "the supported configurations. For EC2 launch type, consider enabling Cluster "
            "Auto Scaling with a capacity provider that adjusts the ASG based on pending "
            "task count. Review placement strategies to confirm they align with the "
            "available instance types. If tasks fail during deployments, check the "
            "deployment circuit breaker configuration and review the minimum healthy "
            "percent setting to provide room for new tasks during rolling updates."
        )},
        "category": {"S": "troubleshooting"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-28T16:20:00Z"},
    },
    {
        "entity_id": {"S": "DOC-015"},
        "sk": {"S": "METADATA"},
        "title": {"S": "CloudWatch alarm noise reduction"},
        "content": {"S": (
            "Reduce CloudWatch alarm noise by tuning evaluation periods, thresholds, and "
            "metric math expressions. Start by analyzing alarm history to identify alarms "
            "that frequently transition between OK and ALARM states without requiring "
            "human intervention. Increase the evaluation period from one to three data "
            "points to filter transient spikes. Use the M out of N evaluation model where "
            "the alarm triggers only when M of the last N data points breach the threshold. "
            "Apply anomaly detection bands for metrics with variable baselines rather than "
            "static thresholds. Group related alarms into composite alarms that trigger "
            "only when multiple conditions are met simultaneously. For percentage-based "
            "metrics, add a minimum sample size condition using metric math to avoid "
            "alerting on low-traffic periods where a single error produces a 100 percent "
            "error rate. Review alarm actions to confirm they route to the appropriate "
            "notification channel based on severity and time of day."
        )},
        "category": {"S": "troubleshooting"},
        "entity_type": {"S": "document"},
        "created_at": {"S": "2025-01-29T12:10:00Z"},
    },
]


def seed_documents(table_name: str, region: str) -> None:
    """Insert sample documents into the DynamoDB table."""
    dynamodb = boto3.client("dynamodb", region_name=region)

    print(f"Seeding {len(SAMPLE_DOCUMENTS)} documents into table: {table_name}")

    for doc in SAMPLE_DOCUMENTS:
        entity_id = doc["entity_id"]["S"]
        title = doc["title"]["S"]

        try:
            dynamodb.put_item(TableName=table_name, Item=doc)
            print(f"  Inserted: {entity_id} - {title}")
        except Exception as e:
            print(f"  Failed to insert {entity_id}: {e}")

    print(f"\nFinished seeding {len(SAMPLE_DOCUMENTS)} documents.")


def wait_for_embeddings(table_name: str, region: str, timeout: int = 120) -> bool:
    """Poll the table until all documents have embeddings or timeout is reached."""
    dynamodb = boto3.client("dynamodb", region_name=region)

    print(f"\nWaiting for embedding pipeline to generate embeddings (timeout: {timeout}s)...")

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        response = dynamodb.scan(
            TableName=table_name,
            FilterExpression="attribute_not_exists(embedding)",
            Select="COUNT",
        )

        missing_count = response["Count"]

        if missing_count == 0:
            elapsed = int(time.time() - start_time)
            print(f"All embeddings generated successfully in {elapsed} seconds.")
            return True

        elapsed = int(time.time() - start_time)
        print(f"  [{elapsed}s] Waiting... {missing_count} documents still missing embeddings.")
        time.sleep(5)

    print(f"\nTimeout: some documents still missing embeddings after {timeout} seconds.")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Seed sample documents into the DynamoDB vector search table."
    )
    parser.add_argument(
        "--table-name",
        required=True,
        help="Name of the DynamoDB table to seed data into.",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS Region (default: us-east-1).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for embedding pipeline to generate all embeddings.",
    )

    args = parser.parse_args()

    seed_documents(args.table_name, args.region)

    if args.wait:
        success = wait_for_embeddings(args.table_name, args.region)
        if not success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
