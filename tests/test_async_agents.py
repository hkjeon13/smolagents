import argparse
import asyncio
import os
import sys
from smolagents.async_models import AsyncAzureOpenAIServerModel
from smolagents.async_agents import AsyncCodeAgent

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


async def main(args: argparse.Namespace):
    model = AsyncAzureOpenAIServerModel(
        model_id=args.model_id,
        api_key=args.api_key,
        api_version=args.api_version,
        azure_endpoint=args.azure_endpoint
    )

    agent = AsyncCodeAgent(
        tools=[],
        model=model,
        add_base_tools=True,
        max_steps=10
    )

    question = "트럼프가 말하는 상호관세가 뭐야?"
    output = await agent.run(question,stream=False, reset=False)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent")
    parser.add_argument("--backend_name", type=str, default="azure-openai", help="The backend name.")
    parser.add_argument("--model_id", type=str, default="gpt4o", help="The model name.")
    parser.add_argument("--api_key", type=str, default="", help="The api key.")
    parser.add_argument("--api_version", type=str, default="2024-12-01-preview", help="The api version.")
    parser.add_argument("--azure_endpoint", type=str, default="https://ai-tech-openai.openai.azure.com",
                        help="The azure endpoint.")
    parser.add_argument("--agent_status_db_path", type=str, default="agent_status.db", help="The agent status path.")
    parser.add_argument("--naver_client_id", type=str, default="", help="The Naver client ID.")
    parser.add_argument("--naver_client_secret", type=str, default="", help="The Naver client secret.")
    args = parser.parse_args()
    asyncio.run(main(args))
