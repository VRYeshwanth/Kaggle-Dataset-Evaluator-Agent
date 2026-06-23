"""
chat.py

Programmatic interface for the ADK agent.

Used by Streamlit.
"""

import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent.agent import root_agent


APP_NAME = "kaggle_dataset_evaluator"

session_service = InMemorySessionService()

runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)


async def create_session(
    user_id: str = "streamlit_user",
):
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=str(uuid.uuid4()),
    )

    return session.id


async def ask_agent(
    question: str,
    session_id: str,
    user_id: str = "streamlit_user",
):
    response_text = ""

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=question)
                ],
            ),
        ):

            if (
                hasattr(event, "content")
                and event.content
                and event.content.parts
            ):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

        return response_text

    except Exception as e:
        return f"Agent execution failed: {e}"