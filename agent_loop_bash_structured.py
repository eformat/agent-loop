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

bash_tool = {
    "name": "bash",
    "description": "Execute bash commands and return the output",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            }
        },
        "required": ["command"],
        "additionalProperties": False
    }
}

tools = [bash_tool]

class BashToolCall(BaseModel):
    command: str

json_schema = BashToolCall.model_json_schema()

# Function to execute bash commands
def execute_bash(command):
    """Execute a bash command and return a formatted string with the results."""
    # If we have a timeout exception, we'll return an error message instead
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=10
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

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
        You are an expert in composing bash functions.
        You are given a question.
        Based on the question, you will need to make one or more bash function calls to achieve the purpose.
        Make sure to format bash in the command correctly and that any bash is syntactically correct.
        You should not include any other text in the response.
        An example output would be:

        {"command":"echo hello world"}

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

def handle_tool_call(tool_call):
    if tool_call["name"] != "bash":
        raise Exception(f"Unsupported tool: {tool_call['name']}")

    command = tool_call["input"]["command"]
    print(f"Executing bash command: {command}")
    output_text = execute_bash(command)
    print(f"Bash output:\n{output_text}")
    return dict(
        type="tool_result",
        tool_use_id=tool_call["id"],
        content=[dict(
            type="text",
            text=output_text
        )]
    )

if __name__ == "__main__":
    main()


# python -m vllm.entrypoints.openai.api_server --model /home/mike/instructlab/models/Llama-3.2-3B-Instruct-Q8_0.gguf --served-model-name=Llama-3.2-3B-Instruct-Q8_0.gguf --quantization gguf --port 8080 --max-model-len=6000 --enable-auto-tool-choice --tool-call-parser=llama3_json --chat-template=/home/mike/git/vllm/examples/tool_chat_template_llama3.2_json.jinja


# You: say hello world
# Agent:  {"command":"echo 'Hello World'"}
# You: remove a directory called foo
# Agent:  {"command":"rm -r foo"}
# You: tell me the time
# Agent:  {"command":"date"}
# You: write a simple loop
# Agent:  {"command":"for i in {1..5; do echo $i; done;}"}
# You: 
# You: create a function to add a list of numbers together
# Agent:  {"command":"add_numbers() { local sum=0; for num in \"1 2 3 4 5\"; do (( sum += num )); done; echo $sum; }"}

