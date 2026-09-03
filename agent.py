from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from tools import search_and_verify_references
from dotenv import load_dotenv
from autogen_core.tools import FunctionTool
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# The response format for the agent as a Pydantic base model.
class QAItem(BaseModel):
    question: str
    answer: str
    reference: Optional[str] = None  # null if no reference could be verified


class QAList(BaseModel):
    items: List[QAItem]

search_and_verify_tool = FunctionTool(
    search_and_verify_references,
    description="Search for the reference url for each of the questions and verify their relevance.",
    strict=True
)

# Create an agent that uses the OpenAI GPT-4o model.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=API_KEY)
agent = AssistantAgent(
    "qa_generator",
    model_client=model_client,
    tools=[search_and_verify_tool],
    output_content_type=QAList,
    system_message="""
    You are a Question-Answer Generation Agent.
    
    Given a topic, a complexity level (Beginner/Intermediate/Advanced), and a
    question_count, do the following:
    
    1. Generate exactly `question_count` distinct questions on the topic, matching
       the requested complexity.
    2. Write a detailed, accurate answer for each question.
    3. Pass a list of questions to the search_and_verify_tool to find and verify a relevant reference URL for each question.
    4. Never invent or guess a URL.

    If the question_count = 3, create a list of 3 questions and pass it to the search_and_verify_tool. Example input format for the tool: ["Question 1", "Question 2", "Question 3"].
    
    Your final reply's structure (question, answer, reference per item) is
    enforced automatically - just make sure every field is filled in correctly.
    """
)

def generate_qa(topic: str, complexity: str, question_count: int) -> list:
    task = (
        f"Topic: {topic}\n"
        f"Complexity: {complexity}\n"
        f"Number of questions: {question_count}\n"
        "Please generate and verify the Q&A set as instructed."
    )

    return task

async def main():

    sample_input = {
        "topic": "Agentic AI",
        "complexity": "Intermediate",
        "question_count": 3,
    }

    task = generate_qa(
        topic=sample_input["topic"],
        complexity=sample_input["complexity"],
        question_count=sample_input["question_count"],
    )

    result = await agent.run(task=task)
    final_message = result.messages[-1]
    response: QAList = final_message.content

    for item in response.items:
        print("Question:", item.question)
        print("Answer:", item.answer)
        print("Reference:", item.reference)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())