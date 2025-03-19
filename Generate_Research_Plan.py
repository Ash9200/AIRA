import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
import anthropic
import re

# Add these color codes at the beginning of your script
GREEN = '\033[92m'
ORANGE = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Constants
MAX_ITERATIONS = 5
BASE_OUTPUT_DIR = ''

# Pydantic models for structured output
class ResearchStep(BaseModel):
    name: str = Field(..., description="Name of the research step")
    description: str = Field(..., description="Description of the step's goals")

class ResearchPlan(BaseModel):
    steps: List[ResearchStep] = Field(..., description="List of research steps")

class ValidationResult(BaseModel):
    success: bool = Field(..., description="Whether the research plan adequately addresses the research question")
    explanation: str = Field(..., description="Explanation of why the plan is successful or needs improvement")

def load_research_question(file_path: str = "") -> Dict[str, str]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[0]

def generate_research_plan(question: str, explanation: str) -> ResearchPlan:
    prompt = "[prompt placeholder]"

    message = anthropic_client.messages.create(
        model="[model placeholder]",
        max_tokens=4000,
        temperature=0,
        system="You are an AI researcher creating a detailed research plan in JSON format.",
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = message.content[0].text

    try:
        plan_dict = json.loads(response_content)
        return ResearchPlan(**plan_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse the JSON content from the response: {str(e)}")

def validate_research_plan(question: str, explanation: str, plan: ResearchPlan) -> ValidationResult:
    prompt = "[prompt placeholder]"

    completion = client.beta.chat.completions.parse(
        model="[model placeholder]",
        response_format=ValidationResult,
        messages=[
            {"role": "system", "content": "You are an expert research validator."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.parsed

def refine_research_plan(question: str, explanation: str, current_plan: ResearchPlan, feedback: str) -> ResearchPlan:
    prompt = "[prompt placeholder]"

    message = anthropic_client.messages.create(
        model="[model placeholder]",
        max_tokens=4000,
        temperature=0,
        system="You are an AI researcher refining a detailed research plan in JSON format.",
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = message.content[0].text

    try:
        plan_dict = json.loads(response_content)
        return ResearchPlan(**plan_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse the JSON content from the response: {str(e)}")

def save_research_plan(plan: ResearchPlan, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'research_plan.json'), 'w') as f:
        plan_dict = {
            "steps": [
                {
                    "name": step.name,
                    "description": step.description
                }
                for step in plan.steps
            ]
        }
        json.dump(plan_dict, f, indent=2)

def print_color(text, color):
    print(f"{color}{text}{RESET}")

def main():
    # Load the research question
    research_data = load_research_question()
    question = research_data['question']
    explanation = research_data['explanation']

    # Generate initial research plan
    plan = generate_research_plan(question, explanation)
    print_color("\nInitial Research Plan:", GREEN)
    for step in plan.steps:
        print_color(f"\nStep: {step.name}", BLUE)
        print(f"Description: {step.description}")

    for iteration in range(MAX_ITERATIONS):
        print_color(f"\nIteration {iteration + 1}", GREEN)

        # Validate the plan
        validation = validate_research_plan(question, explanation, plan)

        print_color(f"Validation Result: {'Success' if validation.success else 'Failed'}", GREEN if validation.success else RED)
        print(f"Explanation: {validation.explanation}")

        if validation.success:
            print_color("Research plan validated successfully!", GREEN)
            break
        else:
            print_color("Refining the research plan...", GREEN)
            plan = refine_research_plan(question, explanation, plan, validation.explanation)
            print_color("\nRefined Research Plan:", GREEN)
            for step in plan.steps:
                print_color(f"\nStep: {step.name}", BLUE)
                print(f"Description: {step.description}")

    if not validation.success:
        print_color(f"Failed to create a valid research plan after {MAX_ITERATIONS} iterations.", RED)
        return

    # Save the final research plan
    final_output_dir = ''
    save_research_plan(plan, final_output_dir)
    print_color(f"\nFinal research plan saved to {final_output_dir}/research_plan.json", GREEN)

    # Print the final research plan
    print_color("\nFinal Research Plan:", GREEN)
    for step in plan.steps:
        print_color(f"\nStep: {step.name}", BLUE)
        print(f"Description: {step.description}")

    print_color("\nResearch planning process completed successfully!", GREEN)

if __name__ == "__main__":
    main()
