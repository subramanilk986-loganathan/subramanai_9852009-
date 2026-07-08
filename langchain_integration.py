# LangChain integration (requires OPENAI_API_KEY set in env to use OpenAI LLM)
# This file creates a simple Tool wrapper around the prediction function.
# NOTE: To run the LangChain agent you need to `pip install langchain openai` and set OPENAI_API_KEY.
import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from src.predict import predict_heart_disease, parse_input_string

def create_agent():
    # The tool expects a string containing a JSON/dict of features.
    def tool_func(s: str):
        data = parse_input_string(s)
        return str(predict_heart_disease(data))

    heart_tool = Tool(
        name='Heart Disease Predictor',
        func=tool_func,
        description='Input should be a JSON/dict string of patient features. Returns model prediction.'
    )

    # Initialize LLM (requires environment variable OPENAI_API_KEY)
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError('OPENAI_API_KEY not set. Set it in your environment to use the LLM.')

    llm = OpenAI(temperature=0)
    agent = initialize_agent([heart_tool], llm, agent='zero-shot-react-description', verbose=True)
    return agent

if __name__ == '__main__':
    print('This module defines create_agent(). Import and call create_agent() from your app.')
