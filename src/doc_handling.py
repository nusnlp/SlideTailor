
from utils import tenacity
import json
REFINE_TEMPLATE_PATH = "prompts/document_refine.txt"
# CONDITIONAL_REFINE_TEMPLATE_PATH = "prompts/conditional_document_refine.txt"
CONDITIONAL_REFINE_TEMPLATE_PATH = "prompts/conditional_document_refine_test.txt"
PRESENTATION_GUIDELINES_TEMPLATE_PATH = "prompts/conditional_document_refine_gen_guides_v2.txt"
CONDITIONAL_REFINE_WITH_GUIDELINES_TEMPLATE_PATH = "prompts/conditional_document_refine_with_guidelines_v4.txt"

@tenacity
def refine_document(language_model, markdown_document: str):
    """
    Use your refine prompt to convert raw parsed PDF text
    into a structured JSON (doc_json).
    """
    with open(REFINE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        refine_template = f.read()

    prompt = refine_template.replace("{{markdown_document}}", markdown_document)
    doc_json = language_model(prompt, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json

@tenacity
def conditional_refine_document(language_model, reference_pdf: str, reference_slide: str, markdown_document: str):
    """
    Use your refine prompt to convert raw parsed PDF text
    into a structured JSON (doc_json).
    """
    with open(CONDITIONAL_REFINE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        refine_template = f.read()

    refine_template = refine_template.replace("{{target paper}}", markdown_document)
    refine_template = refine_template.replace("{{reference content pdf}}", reference_pdf)
    refine_template = refine_template.replace("{{reference content slide}}", reference_slide)
    doc_json = language_model(refine_template, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json


def generate_preference_presentation_guidelines(language_model, reference_pdf: str, reference_slide: str):
    """
    Generate presentation guidelines from a reference pdf and slide.
    """
    with open(PRESENTATION_GUIDELINES_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        presentation_guidelines_template = f.read()

    presentation_guidelines_template = presentation_guidelines_template.replace("{{reference content pdf}}", reference_pdf)
    presentation_guidelines_template = presentation_guidelines_template.replace("{{reference content slide}}", reference_slide)
    doc_json = language_model(presentation_guidelines_template, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json


def conditional_refine_document_with_guidelines(language_model, markdown_document: str, guidelines: str):
    """
    Use your refine prompt to convert raw parsed PDF text
    into a structured JSON (doc_json).
    """
    with open(CONDITIONAL_REFINE_WITH_GUIDELINES_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        refine_template = f.read()

    refine_template = refine_template.replace("{{target paper}}", markdown_document)
    refine_template = refine_template.replace("{{user preference guidelines}}", json.dumps(guidelines))
    doc_json = language_model(refine_template, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json
