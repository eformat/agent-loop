#!/usr/bin/env -S uv run --script
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

def main():
    try:
        print("\n=== LLM Agent Loop with Bash Tools ===\n")
        print("Type 'exit' to end the conversation.\n")
        loop(LLM("Llama-3.2-3B-Instruct-Q8_0.gguf"))
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")

def loop(llm):
    msg = user_input()
    while True:
        output = llm(msg)
        print("Agent: ", output)
        msg = user_input()

weather_tool = {
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}

tools = [weather_tool]

class Temp(str, Enum):
    celcius = "celcius"
    farenheit = "farenheit"
    kelvin = "kelvin"

class WeatherToolCall(BaseModel):
    function: str
    city: str
    metric: Temp

json_schema = WeatherToolCall.model_json_schema()

def user_input():
    x = input("You: ")
    if x.lower() in ["exit", "quit"]:
        print("\nExiting agent loop. Goodbye!")
        raise SystemExit(0)
    return [{"type": "text", "text": x}]

class LLM:
    def __init__(self, model):
        #if "OPENAI_API_KEY" not in os.environ:
        #    raise ValueError("OPENAI_API_KEY environment variable not found.")
        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")
        self.model = model
        self.messages = []
        # Llama3.2 prompting format
        # https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
        self.system_prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in composing functions. You are given a question and a set of possible functions.
        Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
        also point it out. You should only return the function or tool call.
        Use the exact function call or tool name in the output for function.
        You should not include any other text in the response.
        An example output would be:

        {"function":"get_weather","city":"Singapore","metric":"celcius"}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        """
        self.tools = tools
        self.messages.append({"role": "system", "content": self.system_prompt})

    def __call__(self, content):
        self.messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=5000,
            messages=self.messages,
            extra_body={"guided_json": json_schema},
#            tools=self.tools,
            stream=False
        )
        output_text = ""

        for content in response.choices[0].message.content:
            output_text += content

        return output_text

if __name__ == "__main__":
    main()


# python -m vllm.entrypoints.openai.api_server --model /home/mike/instructlab/models/Llama-3.2-3B-Instruct-Q8_0.gguf --served-model-name=Llama-3.2-3B-Instruct-Q8_0.gguf --quantization gguf --port 8080 --max-model-len=6000 --enable-auto-tool-choice --tool-call-parser=llama3_json --chat-template=/home/mike/git/vllm/examples/tool_chat_template_llama3.2_json.jinja

# You: what is the weather in Brisbane?
# Agent:  {"function":"get_weather","city":"Brisbane","metric":"farenheit"}
# You: what is the weather in Brisbane in celcius?
# Agent:  {"function":"get_weather","city":"Brisbane","metric":"celcius"}
# You: what is the weather in Brisbane in farenheight?
# Agent:  {"function":"get_weather","city":"Brisbane","metric":"farenheit"}
# You: what is the weather in Brisbane in bongos?
# Agent:  {"function":"get_weather","city":"Brisbane","metric":"farenheit"}
# You: what is the weather in Brisbane in kelvin?
# Agent:  {"function":"get_weather","city":"Brisbane","metric":"kelvin"}
