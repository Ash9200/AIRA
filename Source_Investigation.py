import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import PyPDF2
import anthropic
from openai import OpenAI

class Config:
    PAPERS_FOLDER = ""
    OUTPUT_FOLDER = ""
    ORIGINAL_RESEARCH_FILE = ""
    EXPERIMENT_FOLDER = ""
    NUM_QUESTIONS = 12

class Utils:
    @staticmethod
    def read_pdf(file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    @staticmethod
    def save_json(data: Dict, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def ensure_directory_exists(directory: str):
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def concatenate_paper_summaries(summaries: List[Dict], filenames: List[str], output_file: str):
        concatenated = {"summaries": []}
        for i, (summary, filename) in enumerate(zip(summaries, filenames), start=1):
            summary["document_number"] = i
            summary["filename"] = filename
            concatenated["summaries"].append(summary)
        Utils.save_json(concatenated, output_file)

class PaperSummarySchema(BaseModel):
    citation: str = Field(..., description="The MLA citation for the paper")
    summary: str = Field(..., description="A concise summary of the paper, similar to an abstract")
    relevance: str = Field(..., description="A paragraph detailing the relevance of the paper to the research hypothesis")

class InvestigationResultSchema(BaseModel):
    question: str = Field(..., description="The investigative question being answered")
    answer: str = Field(..., description="A detailed answer to the investigative question")
    relevant_excerpt: str = Field(..., description="The relevant excerpt from the source paper used to answer this question")

class ExperimentSummary(BaseModel):
    interpretation: str = Field(..., description="A detailed interpretation of the experiment results in the context of the research hypothesis")
    implications: str = Field(..., description="The implications of the results for how the paper should be structured")
    key_findings: List[str] = Field(..., description="A list of key findings from the experiment")
    limitations: List[str] = Field(..., description="A list of limitations or potential issues with the experiment")
    methodology: str = Field(..., description="A detailed description of the methodology used in the experiment")

class PaperSummary:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def process_paper(self, paper_content: str, research_hypothesis: str) -> PaperSummarySchema:
        prompt = "[prompt placeholder]"

        completion = self.client.beta.chat.completions.parse(
            model="[model placeholder]",
            messages=[
                {"role": "system", "content": "[prompt placeholder]"},
                {"role": "user", "content": prompt}
            ],
            response_format=PaperSummarySchema
        )

        return completion.choices[0].message.parsed

class InvestigativeInstance:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.conversation_history = []

    def generate_questions(self, research_hypothesis: str, paper_summaries: Dict, num_questions: int = Config.NUM_QUESTIONS) -> List[str]:
        questions = []
        initial_prompt = "[prompt placeholder]"
        self.conversation_history = [{"role": "user", "content": initial_prompt}]

        for i in range(num_questions):
            response = self.client.messages.create(
                model="[model placeholder]",
                max_tokens=8192,
                messages=self.conversation_history
            )
            question = response.content[0].text.strip()
            questions.append(question)
            self.conversation_history.append({"role": "assistant", "content": question})
            self.conversation_history.append({"role": "user", "content": "Continue"})
            print(f"Generated question {i+1}/{num_questions}")
        return questions

class SourcePaperRepresentative:
    def __init__(self, paper_content: str):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.paper_content = paper_content

    def answer_question(self, question: str, research_hypothesis: str) -> InvestigationResultSchema:
        prompt = "[prompt placeholder]"
        completion = self.client.beta.chat.completions.parse(
            model="[model placeholder]",
            messages=[
                {"role": "system", "content": "[prompt placeholder]"},
                {"role": "user", "content": prompt}
            ],
            response_format=InvestigationResultSchema
        )
        return completion.choices[0].message.parsed

class ExperimentSummarizationInstance:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def summarize_experiment(self, research_hypothesis: str) -> Optional[ExperimentSummary]:
        experiment_folder = Config.EXPERIMENT_FOLDER
        results_file = os.path.join(experiment_folder, "Total_Results.json")

        if not os.path.exists(results_file):
            print("No Total_Results.json found in Experiment folder. Skipping experiment summarization.")
            return None

        with open(results_file, 'r', encoding='utf-8') as file:
            results_text = file.read()

        prompt = "[prompt placeholder]"

        completion = self.client.beta.chat.completions.parse(
            model="[model placeholder]",
            messages=[
                {"role": "system", "content": "[prompt placeholder]"},
                {"role": "user", "content": prompt}
            ],
            response_format=ExperimentSummary
        )

        return completion.choices[0].message.parsed
