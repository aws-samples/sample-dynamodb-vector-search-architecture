#!/usr/bin/env python3
"""Test the deployed Bedrock agent with sample queries.

Usage:
    python scripts/test_agent.py --agent-id <AGENT_ID> --agent-alias-id <ALIAS_ID> --region us-east-1
"""

import argparse
import uuid

import boto3


SAMPLE_QUERIES = [
    "What's our retry strategy for payment failures?",
    "Find documents about database migration architecture",
    "Get details for document DOC-001",
]


def invoke_agent(client, agent_id: str, agent_alias_id: str, query: str) -> str:
    """Invoke the Bedrock agent and return the full response text."""
    session_id = str(uuid.uuid4())

    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        inputText=query,
    )

    response_text = ""
    for event in response["completion"]:
        if "chunk" in event:
            chunk_bytes = event["chunk"].get("bytes", b"")
            response_text += chunk_bytes.decode("utf-8")

    return response_text, session_id


def main():
    parser = argparse.ArgumentParser(
        description="Test the deployed Bedrock agent with sample queries."
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Bedrock agent ID.",
    )
    parser.add_argument(
        "--agent-alias-id",
        required=True,
        help="Bedrock agent alias ID.",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS Region (default: us-east-1).",
    )

    args = parser.parse_args()

    client = boto3.client("bedrock-agent-runtime", region_name=args.region)

    print("=" * 70)
    print("Bedrock Agent Test")
    print(f"Agent ID: {args.agent_id}")
    print(f"Alias ID: {args.agent_alias_id}")
    print(f"Region:   {args.region}")
    print("=" * 70)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 70}")

        try:
            response_text, session_id = invoke_agent(
                client, args.agent_id, args.agent_alias_id, query
            )
            print(f"Session ID: {session_id}")
            print(f"\nResponse:\n{response_text}")
        except Exception as e:
            print(f"Error invoking agent: {e}")

    print(f"\n{'=' * 70}")
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
