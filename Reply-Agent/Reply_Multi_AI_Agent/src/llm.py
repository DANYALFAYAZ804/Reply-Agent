import os
from langchain_openai import ChatOpenAI


def get_model(
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_analyst_model() -> ChatOpenAI:
    return get_model(temperature=0.2, max_tokens=600)


def get_detector_model() -> ChatOpenAI:
    return get_model(temperature=0.1, max_tokens=600)


def get_strategist_model() -> ChatOpenAI:
    return get_model(temperature=0.5, max_tokens=800)


def get_coordinator_model() -> ChatOpenAI:
    return get_model(temperature=0.3, max_tokens=800)
