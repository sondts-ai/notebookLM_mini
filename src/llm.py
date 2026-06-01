from functools import lru_cache
from langchain_core.messages import (
    HumanMessage
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI
)
from config import Settings

def _build_hf_local():
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

    tokenizer=AutoTokenizer.from_pretrained(Settings.hf_model)
    model=AutoModelForCausalLM.from_pretrained(Settings.hf_model,dtype=torch.bfloat16)

    text_gen=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
    )

    text_gen.generation_config.max_new_token=Settings.hf_max_new_token
    text_gen.generation_config.do_sample=Settings.llm_temperature>0

    return ChatHuggingFace(pipeline=text_gen)


def _build_gemini():
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )  

def _build_vllm():
    return ChatOpenAI(
        model=settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )

@lru_cache(maxsize=4)
def get_llm(provider=None):
    provider=(provider or settings.llm_provider)

    if provider=="hf_local":
        return _build_hf_local()
    if provider=="gemini":
        return _build_gemini()
    if provider=="vllm":
        return _build_vllm()

    raise ValueError(f"Unknown llm provider `provider` ")
    
def invoke_llm(prompt, provider=None):
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content,str) else str(response.content)