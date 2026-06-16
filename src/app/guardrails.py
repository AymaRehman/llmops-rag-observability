import os
import re

from fastapi import HTTPException
from langchain_ollama import OllamaLLM

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

guard_llm = OllamaLLM(base_url=OLLAMA_URL, model="llama-guard3:1b")

FORBIDDEN_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"you are now DAN",
    r"drop database",
    r"delete from",
    r"hack into",
    r"disregard (all )?prior",
    r"forget (everything|all instructions)",
    r"you are now in developer mode",
    r"jailbreak",
    r"override rules",
    r"do anything now",
    r"repeat your instructions",
    r"output your system prompt",
    r"show me your base prompt",
    r"what are your guidelines",
    r"\[system\]",
    r"<\|im_start\|>",
    r"```system",
    r"eval\(",
    r"exec\(",
    r"select \* from",
]

def check_regex_patterns(prompt: str):
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise HTTPException(
                status_code=400, 
                detail="Security Exception: Forbidden regex pattern detected."
            )

def check_llama_guard(prompt: str):
    response = guard_llm.invoke(prompt).strip().lower()

    if response.startswith("unsafe"):
        raise HTTPException(
            status_code=400,
            detail=f"Security Exception: Prompt flagged as unsafe by Llama Guard ({response})."
        )

def validate_user_input(prompt: str) -> str:
    check_regex_patterns(prompt)
    check_llama_guard(prompt)
    
    return prompt