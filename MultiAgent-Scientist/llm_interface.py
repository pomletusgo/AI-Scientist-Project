"""Qwen LLM 接口层 —— 支持 DashScope API 和 OpenAI 兼容模式"""

import os
from openai import OpenAI


def create_qwen_client(mode="dashscope"):
    """创建 Qwen 客户端

    mode="dashscope":  DashScope OpenAI 兼容模式（推荐）
    mode="native":     DashScope 原生 API
    mode="local":      本地部署（vLLM/Ollama）
    """
    if mode == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ), "qwen-plus"

    elif mode == "native":
        import dashscope
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = api_key
        return dashscope, "qwen-plus"

    elif mode == "local":
        base_url = os.environ.get("QWEN_LOCAL_URL", "http://localhost:8000/v1")
        return OpenAI(api_key="not-needed", base_url=base_url), "qwen-local"

    else:
        raise ValueError(f"Unknown mode: {mode}")


def call_qwen(messages, client, model, temperature=0.7, max_tokens=4096, mode="dashscope"):
    """统一调用接口"""
    if mode == "native":
        from dashscope import Generation
        response = Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format="message",
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"DashScope error: {response.code} {response.message}")

    else:
        # OpenAI 兼容模式（dashscope + local）
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
