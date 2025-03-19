import json
import os
from openai import OpenAI
import anthropic
from PyPDF2 import PdfReader
from typing import Dict, List, Optional

class Config:
    RESEARCH_QUESTION_PATH = ""
    OUTLINE_PATH = ""
    PAPER_SUMMARIES_PATH = ""
    EXPERIMENT_SUMMARY_PATH = ""
    EXPERIMENT_RESULTS_PATH = ""
    PAPERS_FOLDER = ""
    OUTPUT_FILE = ""

class Utils:
    @staticmethod
    def read_pdf(file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
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
    def extract_research_info(input_file: str) -> Dict:
        data = Utils.load_json(input_file)
        if not data or not isinstance(data, list) or len(data) == 0:
            raise ValueError("Invalid input JSON structure")
        
        research_info = data[0]  # Assuming the first item contains the research info
        return {
            "hypothesis": research_info.get("hypothesis", ""),
            "explanation": research_info.get("explanation", ""),
            "domain": research_info.get("domain", "")
        }

    @staticmethod
    def ensure_directory_exists(directory: str):
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def load_outline() -> Dict:
        return Utils.load_json(Config.OUTLINE_PATH)

    @staticmethod
    def load_paper_summaries() -> Dict:
        return Utils.load_json(Config.PAPER_SUMMARIES_PATH)

    @staticmethod
    def load_experiment_summary() -> Optional[Dict]:
        if os.path.exists(Config.EXPERIMENT_SUMMARY_PATH):
            return Utils.load_json(Config.EXPERIMENT_SUMMARY_PATH)
        return None

    @staticmethod
    def load_experiment_results() -> Optional[Dict]:
        if os.path.exists(Config.EXPERIMENT_RESULTS_PATH):
            return Utils.load_json(Config.EXPERIMENT_RESULTS_PATH)
        return None

class ResearchPaperWriter:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.API_KEY)
        self.research_hypothesis = Utils.extract_research_info(Config.RESEARCH_QUESTION_PATH)
        self.outline = Utils.load_outline()
        self.paper_summaries = Utils.load_paper_summaries()
        self.experiment_summary = Utils.load_experiment_summary()
        self.experiment_results = Utils.load_experiment_results()
        self.written_content = ""

    def extract_pdf_content(self, filename: str) -> str:
        path = os.path.join(Config.PAPERS_FOLDER, filename)
        return Utils.read_pdf(path)

    def get_relevant_papers(self, relevant_documents: List[str]) -> Dict[str, str]:
        relevant_papers = {}
        for summary in self.paper_summaries["summaries"]:
            doc_number = str(summary["document_number"])
            if doc_number in relevant_documents:
                citation = summary.get("citation")
                content = self.extract_pdf_content(summary["filename"])
                relevant_papers[citation] = content
        return relevant_papers

    def write_section(self, section: Dict) -> str:
        relevant_papers = self.get_relevant_papers(section["relevant_documents"])
        
        prompt = "[prompt placeholder]"

        message = self.client.messages.create(
            model="[model placeholder]",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content
    
    def write_paper(self):
        Utils.ensure_directory_exists(os.path.dirname(Config.OUTPUT_FILE))
        
        for section in self.outline["sections"]:
            print(f"Writing section: {section['section_title']}")

            section_content = self.write_section(section)

            # Update the written_content
            self.written_content += f"\n\n# {section['section_title']}\n\n{section_content}\n\n"
            
            # Append the new section to the output file
            with open(Config.OUTPUT_FILE, "a", encoding='utf-8') as f:
                f.write(f"# {section['section_title']}\n\n{section_content}\n\n")
            
            print(f"Completed section: {section['section_title']}")

        print(f"Research paper writing completed. Output saved to {Config.OUTPUT_FILE}")
        
if __name__ == "__main__":
    writer = ResearchPaperWriter()
    writer.write_paper()
