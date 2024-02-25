from tests.crewai_interpreter import Agent, Task, Crew, Process
from interpreter import interpreter
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# 1. Configuration and Tools
llm = Ollama(model="mistral")
interpreter.auto_run = True
interpreter.llm.model = "openai/mistral"
interpreter.llm.api_base = "http://localhost:11434/v1"
interpreter.llm.api_key = "fake_key"
interpreter.offline = True
interpreter.llm.max_tokens = 1000
interpreter.llm.max_retries = 20
interpreter.llm.context_window = 3000

class CLITool:
    @tool("Executor")
    def execute_cli_command(command: str):
        """Create and Execute code using Open Interpreter."""
        result = interpreter.chat(command)
        return result

# 2. Creating an Agent for CLI tasks
cli_agent = Agent(
    role='Software Engineer',
    goal='Always use Executor Tool. Ability to perform CLI operations, write programs and execute using Exector Tool',
    backstory='Expert in command line operations, creating and executing code.',
    tools=[CLITool.execute_cli_command],
    verbose=True,
    llm=llm 
)

# 3. Defining a Task for CLI operations
cli_task = Task(
    description='Identify the OS and then empty my recycle bin',
    agent=cli_agent,
    tools=[CLITool.execute_cli_command]
)

# 4. Creating a Crew with CLI focus
cli_crew = Crew(
    agents=[cli_agent],
    tasks=[cli_task],
    process=Process.sequential,
    manager_llm=llm
)

# 5. Run the Crew

import gradio as gr

def cli_interface(command):
    cli_task.description = command  
    result = cli_crew.kickoff()
    return result

iface = gr.Interface(
    fn=cli_interface, 
    inputs=gr.Textbox(lines=2, placeholder="What action to take?"), 
    outputs="text",
    title="CLI Command Executor",
    description="Execute CLI commands via a natural language interface."
)

iface.launch()

