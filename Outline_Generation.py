import json
import os
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI

class Config:
    OUTPUT_FOLDER = ""
    ORIGINAL_RESEARCH_FILE = ""

class Utils:
    @staticmethod
    def load_json(file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_text(data: str, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)

    @staticmethod
    def ensure_directory_exists(directory: str):
        os.makedirs(directory, exist_ok=True)

# Step 1: Define the structured output schema
class Section(BaseModel):
    section_title: str
    objectives: List[str]
    content: str
    relevant_documents: List[int]
    source_document_usage: str
    experimental: bool 

class ResearchOutline(BaseModel):
    sections: List[Section]

class OutlineGenerationInstance:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate_outline(self, original_research: dict, paper_summaries: dict, 
                         investigation_results: dict, experiment_summary: Optional[dict]) -> ResearchOutline:
        
        # Step 2: Prompt construction
        prompt = "[prompt placeholder]"
        
        # Step 3: Structured response using response_format
        completion = self.client.beta.chat.completions.parse(
            model="[model placeholder]",
            messages=[{"role": "user", "content": prompt}],
            response_format=ResearchOutline
        )
        
        return completion.choices[0].message.parsed  # Parsed output as per schema

class OutlineGenerator:
    def __init__(self):
        self.outline_generator = OutlineGenerationInstance()

    def process(self):
        Utils.ensure_directory_exists(Config.OUTPUT_FOLDER)

        # Load the necessary files
        print("Loading original research info...")
        original_research = Utils.load_json(Config.ORIGINAL_RESEARCH_FILE)

        print("Loading paper summaries...")
        paper_summaries_file = os.path.join(Config.OUTPUT_FOLDER, "paper_summaries.json")
        paper_summaries = Utils.load_json(paper_summaries_file)

        print("Loading investigation results...")
        investigation_results_file = os.path.join(Config.OUTPUT_FOLDER, "investigation_results.json")
        investigation_results = Utils.load_json(investigation_results_file)

        experiment_summary = None
        experiment_summary_file = os.path.join(Config.OUTPUT_FOLDER, "Experiment_Summarization.json")
        if os.path.exists(experiment_summary_file):
            print("Loading experiment summary...")
            experiment_summary = Utils.load_json(experiment_summary_file)

        print("Generating final outline...")
        outline = self.outline_generator.generate_outline(
            original_research, paper_summaries, investigation_results, experiment_summary
        )

        final_outline_file = os.path.join(Config.OUTPUT_FOLDER, "Final_Outline.json")
        Utils.save_text(json.dumps(outline.dict(), indent=2), final_outline_file)
        print("Final outline saved.")

if __name__ == "__main__":
    generator = OutlineGenerator()
    generator.process()
