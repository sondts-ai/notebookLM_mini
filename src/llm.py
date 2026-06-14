from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI

from src.config import Settings


settings = Settings()


def _build_hf_local():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)

    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_model,
        torch_dtype=torch.bfloat16,
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=settings.llm_temperature > 0,
        temperature=settings.llm_temperature,
    )

    return ChatHuggingFace(llm=text_gen)


def _build_gemini():
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm():
    return ChatOpenAI(
        model=settings.hf_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    provider = provider or settings.llm_provider

    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    if provider == "vllm":
        return _build_vllm()

    raise ValueError(f"Unknown llm provider `{provider}`")


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    response = get_llm(provider=provider).invoke(
        [HumanMessage(content=prompt)]
    )
    return response.content if isinstance(response.content, str) else str(response.content)