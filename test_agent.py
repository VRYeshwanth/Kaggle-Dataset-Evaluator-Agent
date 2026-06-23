import asyncio

from agent.chat import (
    create_session,
    ask_agent
)


async def main():
    session_id = await create_session()

    response = await ask_agent(
        "Analyze data/sample.csv",
        session_id,
    )

    print(response)


asyncio.run(main())