#!/usr/bin/env -S uv run --script
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from openai import OpenAI 

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
        also point it out. You should only return the function call in tools call sections.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.

        Here is a list of functions in JSON format that you can invoke.
        [
            {
                "name": "get_weather",
                "description": "Get weather info for places",
                "parameters": {
                    "type": "dict",
                    "required": [
                        "city"
                    ],
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the weather for"
                        },
                        "metric": {
                            "type": "string",
                            "description": "The metric for weather. Options are: celsius, fahrenheit",
                            "default": "celsius"
                        }
                    }
                }
            }
        ]<|eot_id|><|start_header_id|>user<|end_header_id|>
        """
        self.tools = tools
        self.messages.append({"role": "system", "content": self.system_prompt})

    def __call__(self, content):
        self.messages.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=5000,
            messages=self.messages,
#            tools=self.tools,
            stream=False
        )
        output_text = ""

        for content in response.choices[0].message.content:
            output_text += content

        return output_text

if __name__ == "__main__":
    main()


# You: what is the weather in Brisbane?
# Agent:  [get_weather(city="Brisbane")]
# You: what is the weather in Brisbane in celcius?
# Agent:  [get_weather(city="Brisbane", metric="celsius")]
# You: what is the weather in Brisbane in farenheight?
# Agent:  [get_weather(city='Brisbane', metric='fahrenheit')]


        # - python
        # - -m
        # - vllm.entrypoints.openai.api_server
        # - --port=8080
        # - --model=/mnt/models/Llama-3.2-3B-Instruct-Q8_0.gguf
        # - --served-model-name=Llama-3.2-3B-Instruct-Q8_0.gguf
        # - --quantization=gguf
        # - --max-model-len=131072
        # - --enforce-eager
        # - --enable-auto-tool-choice
        # - --tool-call-parser=llama3_json
        # - --chat-template=/app/data/template/tool_chat_template_llama3.2_json.jinja
