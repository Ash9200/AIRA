import json
import os
import traceback
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
import sys
import io
import importlib.util
import anthropic
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

# Constants
BASE_OUTPUT_DIR = ''

class ExperimentResult(BaseModel):
    output: Union[Dict[str, Any], None] = Field(None, description="The output returned by the main function of the experiment")
    success: bool = Field(..., description="A boolean indicating whether the experiment succeeded (True) or failed (False)")
    error_message: Union[str, None] = Field(None, description="If the experiment failed, a string describing the error")
    error_traceback: Union[str, None] = Field(None, description="If the experiment failed, the full error traceback")

    class Config:
        extra = "forbid"

class StepCode(BaseModel):
    code: str = Field(..., description="The complete Python code for this step, including functions and test script")

    class Config:
        extra = "forbid"

class CodeRefinement(BaseModel):
    refined_code: str = Field(..., description="The refined Python code after addressing feedback or errors")
    explanation: str = Field(..., description="Explanation of the changes made to address the feedback or errors")

    class Config:
        extra = "forbid"

class ValidationResult(BaseModel):
    success: bool = Field(..., description="A boolean indicating whether the experiment truly succeeded in addressing the research question")
    explanation: str = Field(..., description="Detailed explanation of the validation decision and guidance for improvement if needed")

    class Config:
        extra = "forbid"

def load_input_json(research_question_path="", research_plan_path=""):
    with open(research_question_path, 'r') as f:
        research_data = json.load(f)
    question = research_data[0]['question']
    explanation = research_data[0]['explanation']

    with open(research_plan_path, 'r') as f:
        plan_data = json.load(f)
    steps = plan_data['steps']

    return question, explanation, steps

def load_previous_steps(current_step: int):
    context = ""
    for step in range(1, current_step):
        step_dir = os.path.join(BASE_OUTPUT_DIR, f"step_{step}")

        # Load code
        code_file = os.path.join(step_dir, f"step_{step}_code.py")
        if os.path.exists(code_file):
            with open(code_file, 'r') as f:
                context += f"Step {step} Code:\n{f.read()}\n\n"

        # Load output
        output_file = os.path.join(step_dir, f"step_{step}_output.txt")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                context += f"Step {step} Output:\n{f.read()}\n\n"

    return context

def generate_step_code(question: str, explanation: str, steps: List[Dict[str, str]], current_step: int, previous_context: str = "") -> StepCode:
    step = steps[current_step]
    prompt = "[prompt placeholder]"

    response = client.chat.completions.create(
        model="[model placeholder]",
        messages=[{"role": "user", "content": prompt}]
    )

    full_response = response.choices[0].message.content
    code_start = full_response.find("CODE:") + len("CODE:")
    code = full_response[code_start:].strip()

    return StepCode(code=code)

class TqdmIO(io.StringIO):
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, s):
        if s.strip():
            tqdm.write(s, file=self.original_stream, end='')
        return super().write(s)

    def flush(self):
        self.original_stream.flush()

def run_experiment(step_num: int, code: str) -> ExperimentResult:
    out_dir = os.path.join(BASE_OUTPUT_DIR, f"step_{step_num}")
    os.makedirs(out_dir, exist_ok=True)

    code_file = os.path.join(out_dir, f"step_{step_num}_code.py")
    output_file = os.path.join(out_dir, f"step_{step_num}_output.txt")

    with open(code_file, "w") as f:
        f.write(code)

    sys.path.insert(0, out_dir)

    tqdm_stdout = TqdmIO(sys.stdout)
    tqdm_stderr = TqdmIO(sys.stderr)

    try:
        spec = importlib.util.spec_from_file_location(f"step_{step_num}_code", code_file)
        code_module = importlib.util.module_from_spec(spec)

        tqdm.write(f"\nExecuting Step {step_num}:")

        with redirect_stdout(tqdm_stdout), redirect_stderr(tqdm_stderr):
            spec.loader.exec_module(code_module)
            output = code_module.main()

        with open(output_file, 'w') as f:
            f.write("Captured Output:\n")
            f.write(tqdm_stdout.getvalue())
            f.write(tqdm_stderr.getvalue())
            f.write("\nReturned Output:\n")
            if isinstance(output, dict):
                json.dump(output, f, indent=2)
            else:
                f.write(str(output))

        return ExperimentResult(output=output, success=True, error_message=None, error_traceback=None)
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        tqdm.write(f"Error occurred: {error_message}")
        tqdm.write(error_traceback)
        return ExperimentResult(output=None, success=False, error_message=error_message, error_traceback=error_traceback)
    finally:
        sys.path.pop(0)

def validate_experiment(question: str, explanation: str, current_code: str, experiment_output: Dict[str, Any], steps: List[Dict[str, str]], current_step: int) -> ValidationResult:
    prompt = "[prompt placeholder]"

    message = anthropic_client.messages.create(
        model="[model placeholder]",
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = message.content[0].text
    success_start = response_content.find("SUCCESS:") + len("SUCCESS:")
    explanation_start = response_content.find("EXPLANATION:")

    success_str = response_content[success_start:explanation_start].strip().lower()
    success = success_str == "true"
    explanation = response_content[explanation_start + len("EXPLANATION:"):].strip()

    return ValidationResult(success=success, explanation=explanation)

def refine_experiment_code(current_code: str, feedback: str, error_traceback: str, question: str, explanation: str, steps: List[Dict[str, str]], current_step: int, human_feedback: str = "") -> CodeRefinement:
    prompt = "[prompt placeholder]"

    message = anthropic_client.messages.create(
        model="[model placeholder]",
        max_tokens=4000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = message.content[0].text
    refined_code_start = response_content.find("REFINED_CODE:") + len("REFINED_CODE:")
    explanation_start = response_content.find("EXPLANATION:")

    refined_code = response_content[refined_code_start:explanation_start].strip()
    explanation = response_content[explanation_start + len("EXPLANATION:"):].strip()

    return CodeRefinement(refined_code=refined_code, explanation=explanation)

def main():
    question, explanation, steps = load_input_json()

    starting_step = int(input(f"Enter the starting step number (1-{len(steps)}): "))

    for step_num in range(starting_step - 1, len(steps)):
        step = steps[step_num]
        print(f"\nStep {step_num + 1}: {step['name']}")

        previous_context = load_previous_steps(step_num + 1) if step_num > 0 else ""

        step_code = generate_step_code(question, explanation, steps, step_num, previous_context)

        results = run_experiment(step_num + 1, step_code.code)

        if not results.success:
            refinement = refine_experiment_code(step_code.code, results.error_message, results.error_traceback, question, explanation, steps, step_num, "")
            step_code.code = refinement.refined_code
            print(f"  Code refined. Explanation: {refinement.explanation}")

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
