import glob
import os
import json
from jinja2 import Template
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from marker.models import create_model_dict
import torch
from openai import OpenAI

# torch.set_num_threads(1)  # for macos, https://github.com/apple/ml-stable-diffusion/issues/8

os.sys.path.append('./src')

from presentation import Presentation
from utils import Config, pptx_to_pdf, ppt_to_images
from llms import LLM, setup_models
from pdf_parsing import parse_pdf, parsing_pdf_with_caption
from stage_modules import stage_reference_document_parsing


#each eval returns a score and a reason
def eval_aesthetic_quality(eval_dir: str, slide_images: list, vision_model: LLM, use_cache: bool=True):
    style_prompt = open("prompts/evaluation/ref_free/ppteval_aesthetic_quality.txt", "r").read()
    slide_scores = []
    slide_descriptions = []
        
    for slide_image in slide_images:
        slide_name = os.path.basename(slide_image)
        print(f"[INFO] Evaluating style of slide: {slide_name}")
    
        # Create evaluation results file path
        slide_eval_path = os.path.join(eval_dir, "aesthetic_quality", f"{slide_name.replace('.jpg', '.json')}")
        os.makedirs(os.path.dirname(slide_eval_path), exist_ok=True)
        
        if not os.path.exists(slide_eval_path) or not use_cache:
            # Direct evaluation of style using unified prompt
            style_eval = vision_model(style_prompt, [slide_image], return_json=True)
            with open(slide_eval_path, "w", encoding="utf-8") as f:
                json.dump({"style": style_eval}, f, indent=4)
        else:
            # Load existing evaluations
            with open(slide_eval_path, "r+", encoding="utf-8") as f:
                eval_data = json.load(f)
                if "style" not in eval_data:
                    style_eval = vision_model(style_prompt, [slide_image], return_json=True)
                    f.seek(0)
                    json.dump(eval_data | {"style": style_eval}, f, indent=4)
                    f.truncate()
                else:
                    style_eval = eval_data['style']
        if "score" in style_eval:
            slide_scores.append(style_eval["score"])
            slide_descriptions.append(f"Score: {style_eval['score']}. Reason: {style_eval.get('reason', 'No description provided')}")        
        
    avg_score = sum(slide_scores) / len(slide_scores) if slide_scores else 0.0
    return avg_score, slide_descriptions

#each eval returns a score and a reason
def eval_content_informativeness(target_ppt_dir: str, target_doc_path: str, vision_model: LLM, language_model: LLM, marker_model: dict, slide_images: list, eval_dir: str, use_cache: bool=True):
    
    slide_eval_dir = os.path.join(eval_dir, "content_info")
    os.makedirs(slide_eval_dir, exist_ok=True)

    if os.path.exists(os.path.join(slide_eval_dir, "target_pdf_content.md")):
        with open(os.path.join(slide_eval_dir, "target_pdf_content.md"), "r", encoding="utf-8") as f:
            pdf_content = f.read()
    else:
        # Parse the source document once
        pdf_content = parsing_pdf_with_caption(pdf_path=target_doc_path, parsed_pdf_dir=slide_eval_dir, marker_model=marker_model, vision_model=vision_model, language_model=language_model, use_cache=False)
        with open(os.path.join(slide_eval_dir, "target_pdf_content.md"), "w", encoding="utf-8") as f:
            f.write(pdf_content)
    
    prompt_template = Template(open("prompts/evaluation/ref_free/ppteval_content.txt", "r").read())
    prompt = prompt_template.render(paper=pdf_content)

    slide_scores = []
    slide_descriptions = []
        
    for slide_image in slide_images:
        slide_name = os.path.basename(slide_image)
        print(f"[INFO] Evaluating content and informativeness of slide: {slide_name}")
    
        # Create evaluation results file path, storing in a sub-directory
        slide_eval_path = os.path.join(slide_eval_dir, f"{slide_name.replace('.jpg', '.json')}")
        os.makedirs(os.path.dirname(slide_eval_path), exist_ok=True)
                
        eval_data = {}
        if os.path.exists(slide_eval_path) and use_cache:
            with open(slide_eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

        if "content_informativeness" not in eval_data:
            # This call is per-slide, passing the full paper and one slide image
            combined_eval = vision_model(prompt, [slide_image], return_json=True)
            eval_data["content_informativeness"] = combined_eval
            with open(slide_eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=4)
        
        combined_eval = eval_data["content_informativeness"]
        if "score" in combined_eval:
            slide_scores.append(combined_eval["score"])
            slide_descriptions.append(f"Score: {combined_eval['score']}. Reason: {combined_eval.get('reason', 'No description provided')}")        
        
    avg_score = sum(slide_scores) / len(slide_scores) if slide_scores else 0.0
    
    return avg_score, slide_descriptions


# #each eval returns a score and a reason
# def eval_coherence(language_model: LLM, presentation_outline: dict):
#     coherence_scorer_template = Template(open("prompts/evaluation/ppteval_coherence.txt", "r").read())  
#     # Use the presentation outline as the target flow but drop "layout" and "image" keys
#     # Evaluate logical coherence via the modified presentation outline
#     coherence_prompt = coherence_scorer_template.render(logical_structure=presentation_outline)
#     coherence_eval = language_model(coherence_prompt, return_json=True)
        
#     score = coherence_eval.get('score', 0)
#     feedback = coherence_eval.get('reason', 'No reason provided')
    
#     return score, feedback


def eval_content_structure_similarity(language_model: LLM, sample_pref_guidelines: dict=None, target_presentation_outline: dict=None, use_cache: bool=True):
    target_presentation_flow = target_presentation_outline
    structure_scorer_template = Template(open("prompts/evaluation/ref_based/ppteval_content_structure_similarity.txt", "r").read())
    # Get sample presentation flows + sections + omitted sections
    sample_presentation_flow = sample_pref_guidelines["presentation_guidelines"]
    
    structure_prompt = structure_scorer_template.render(pres_structure=target_presentation_flow, ref_structure=sample_presentation_flow)
    structure_eval = language_model(structure_prompt, return_json=True)
    
    score = structure_eval.get('score', 0)
    feedback = structure_eval.get('reason', 'No reason provided')
    return score, feedback


def get_narrative_flow(narrative_flow: list, language_model: LLM, categories: list):   
    narrative_prompt = Template(open("prompts/evaluation/ppteval_standardize_sections.txt", "r").read())
    standardised_narrative_flow = language_model(narrative_prompt.render(narrative_flow=narrative_flow, categories=categories), return_json=True)
    return standardised_narrative_flow



def eval_auto_structural_similarity(language_model: LLM, target_pres_outline: dict, sample_pref_guidelines: dict, categories: list, use_cache: bool=True):
    
    #combine similar chunks into one
    def _condense_narrative_flow(narrative_list):
        res = []
        curr = None
        for n in narrative_list:
            if curr != n:
                res.append(n)
            curr = n
        return res
    
    sample_flow = sample_pref_guidelines['presentation_guidelines']['narrative_flow_preference']
    target_flow = target_pres_outline['slide_descriptions']

    sample_narrative_list = [x['standard'] for x in get_narrative_flow(sample_flow, language_model, categories)]
    sample_narrative_flow = _condense_narrative_flow(sample_narrative_list)
    target_narrative_list = [x['standard'] for x in get_narrative_flow(target_flow, language_model, categories)]
    target_narrative_flow = _condense_narrative_flow(target_narrative_list)
    
    coverage_similarity_iou = eval_coverage_similarity(target_narrative_flow, sample_narrative_flow, type='iou')
    structural_similarity_ngld = structural_similarity(target_narrative_flow, sample_narrative_flow, type='ngld')
    
    # print("target_narrative_flow:", target_narrative_flow)
    # print("sample_narrative_flow:", sample_narrative_flow)
    # print("coverage_similarity_iou:", coverage_similarity_iou)
    # print("structural_similarity_ngld:", structural_similarity_ngld)
    # import pdb; pdb.set_trace()
    return {'iou' : coverage_similarity_iou, 'ngld' : structural_similarity_ngld}
        
#type either iou or cosine
#cosine doesn't work for now
def eval_coverage_similarity(target_narrative_flow, sample_narrative_flow, type='iou'):
    if type == 'iou':
        target_coverage = set(target_narrative_flow)
        sample_coverage = set(sample_narrative_flow)
        return len(target_coverage.intersection(sample_coverage)) / len(target_coverage.union(sample_coverage))
    else:
        return NotImplementedError

#Takes in a list of ordered categories, i.e. output from classify_slides
#Returns a number between 0 and 1, the lower the score, the more similar the two sequences
def structural_similarity(target_narrative_flow, sample_narrative_flow, type='ngld'):
    #assuming gamma(insert) = gamma(delete) = 1
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
    
    #taken from "A Normalized Levenshtein Distance Metric" by Li Yujian and Liu Bo
    def normalized_levenshtein_distance(s1, s2):
        gld = levenshtein_distance(s1, s2)
        return 2 * gld / (len(s1) + len(s2) + gld)
        
    if type == 'ngld':
        return 1 - normalized_levenshtein_distance(target_narrative_flow, sample_narrative_flow)
    else:
        return NotImplementedError


def eval_template_similarity(vision_model: LLM, slide_images: list, template_images: list, eval_dir: str, use_cache: bool=True):
    prompt_template = Template(open("prompts/evaluation/ref_based/ppteval_template_similarity.txt", "r").read())
    
    num_of_target_slide = len(slide_images)
    num_of_template_slide = len(template_images)
    prompt = prompt_template.render(num_of_target_slide=num_of_target_slide, num_of_template_slide=num_of_template_slide)
    
    all_images = slide_images + template_images
    eval_path = os.path.join(eval_dir, "template_similarity.json")
    
    if os.path.exists(eval_path) and use_cache:
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_result = json.load(f)
    else:
        eval_result = vision_model(prompt, images=all_images, return_json=True, image_first=True)  # too many images, so image_first=True
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4)
            
    score = eval_result.get('score', 0)
    feedback = eval_result.get('reason', 'No reason provided')
    
    return score, feedback

def extract_simple_outline(slides_path, language_model, narrative_categories, eval_base_dir, use_cache: bool=True):
    slide_descr_path = os.path.join(eval_base_dir, "presentation_outline_llm_extracted.json")
    if os.path.exists(slide_descr_path) and use_cache:
        return json.load(open(slide_descr_path, 'r'))
    else:
        if os.path.splitext(slides_path)[1] == ".pptx":
            presentation = Presentation.from_file(slides_path, Config("/tmp"))
            presentation_content = presentation.to_text()
        else:
            raise NotImplementedError
        ppt_extractor = Template(open("prompts/evaluation/ppteval_extract.txt", "r").read())
        extracted_outline = language_model(
            ppt_extractor.render(categories=narrative_categories, presentation=presentation_content),
            return_json=True,
        )
        json.dump(extracted_outline, open(slide_descr_path, "w"), indent=4)
        return extracted_outline

#currently assumes we have pref guidelines
def get_preference_from_pairs(eval_task_name: str, sample_id: str, sample_dir: str, eval_base_dir: str, vision_model: LLM, language_model: LLM, marker_model: dict, use_cache: bool=True):

    sample_cache_dir = os.path.join(eval_base_dir, "sample_pair")
    
    if not os.path.exists(sample_cache_dir):
        os.makedirs(sample_cache_dir, exist_ok=True)
    
    pref_guidelines_path = os.path.join(sample_cache_dir, f"sample_id_{sample_id}_pref_guidelines.json")
    if os.path.exists(pref_guidelines_path) and use_cache:
        return json.load(open(pref_guidelines_path, 'r'))
    
    # Run reference document parsing
    ref_content_pdf = f"{sample_dir}/paper/{sample_id}.pdf"
    ref_content_ppt = f"{sample_dir}/ppt/{sample_id}.pdf"
    pref_guidelines = stage_reference_document_parsing(
        ref_content_pdf,
        ref_content_ppt,
        marker_model,
        vision_model,
        language_model,
        project_id=f"{sample_id}",
        runs_dir=f"{sample_cache_dir}"
    )
    json.dump(pref_guidelines, open(pref_guidelines_path, 'w'), indent=4)
        
    return pref_guidelines


def evaluate_presentation(eval_task_name: str,
                        vision_model: LLM,
                        language_model: LLM,
                        marker_model: dict,
                        target_id: int = None,
                        target_ppt_path: str = None,
                        target_doc_path: str = None,
                        template_ppt_path: str = None,
                        sample_id: str = None,
                        sample_dir: str = None,
                        narrative_categories: list = None) -> dict:
    """
    Evaluate the generated presentation against sample and guidelines
    1. Content and Vision evaluation (per slide)
    2. Logic evaluation (coherence of the entire presentation)
    3. Structure similarity evaluation (comparing to sample)
        
    Args:
        pptx_path: Path to the generated presentation
        ref_ppt_path: Path to sample presentation
        pref_guidelines: Presentation guidelines (for modification)
        original_presentation_outline: Presentation outline
        
    Returns:
        Dict containing evaluation scores and feedback
    """
    
    try:
        
        print(f"[INFO] Evaluating presentation {target_id}: {target_ppt_path}")

        # Create evaluation directory
        target_ppt_dir = os.path.dirname(target_ppt_path)
        eval_base_dir = os.path.join(target_ppt_dir, "evaluation")
        os.makedirs(eval_base_dir, exist_ok=True)
        
        eval_target_id_dir = os.path.join(eval_base_dir, target_id)
        os.makedirs(eval_target_id_dir, exist_ok=True)
        
        # Create images directory for slide evaluation
        slides_image_dir = os.path.join(eval_target_id_dir, "slides_images")
        os.makedirs(slides_image_dir, exist_ok=True)
        if not os.path.exists(slides_image_dir) or len(os.listdir(slides_image_dir)) == 0:
            ppt_to_images(target_ppt_path, slides_image_dir)
        slide_images = sorted(glob.glob(os.path.join(slides_image_dir, "slide_*.jpg")))

        # import pdb; pdb.set_trace()
        
        # Create images directory for template evaluation
        template_image_dir = os.path.join(eval_target_id_dir, "template_images")
        os.makedirs(template_image_dir, exist_ok=True)
        if not os.path.exists(template_image_dir) or len(os.listdir(template_image_dir)) == 0:
            ppt_to_images(template_ppt_path, template_image_dir)
        template_images = sorted(glob.glob(os.path.join(template_image_dir, "slide_*.jpg")))
        
        # Initialize evaluation results dictionary
        results = {
            "scores": {
                "coverage_iou": 0.0,
                "flow_ngld": 0.0,
                "template_layout_similarity": 0.0,
                "content_structure_similarity": 0.0,
                "aesthetic": 0.0,
                "content": 0.0,
                # "coherence": 0.0,
            },
            "aesthetic_feedback": {},
            "content_informativeness_feedback": {},
            "content_structure_similarity_feedback": {},
            "template_similarity_feedback": {},
            # "coherence_feedback": {},
        }
        
        # import pdb; pdb.set_trace()

        # Get target presentation outline, preference guidelines
        target_pres_outline = extract_simple_outline(
            slides_path=target_ppt_path,
            language_model=language_model,
            narrative_categories=narrative_categories,
            eval_base_dir=eval_target_id_dir,
            use_cache=True,
        )
        
        sample_pref_guidelines = get_preference_from_pairs(
            eval_task_name=eval_task_name,
            sample_id=sample_id,
            sample_dir=sample_dir,
            eval_base_dir=eval_base_dir,
            vision_model=vision_model,
            language_model=language_model,
            marker_model=marker_model,
            use_cache=True,
        )

        # get scores and feedback
        automatic_similarity_score = eval_auto_structural_similarity(
            language_model=language_model,
            target_pres_outline=target_pres_outline,
            sample_pref_guidelines=sample_pref_guidelines,
            categories=narrative_categories,
            use_cache=False,
        )
        
        template_similarity_score, template_similarity_feedback = eval_template_similarity(
            vision_model=vision_model,
            slide_images=slide_images,
            template_images=template_images,
            eval_dir=eval_target_id_dir,
            use_cache=False,
        )
        
        content_structure_similarity_score, content_structure_similarity_feedback = eval_content_structure_similarity(
            language_model=language_model,
            sample_pref_guidelines=sample_pref_guidelines,
            target_presentation_outline=target_pres_outline,
            use_cache=False,
        )
        
        aesthetic_quality_score, aesthetic_quality_feedback = eval_aesthetic_quality(
            eval_dir=eval_target_id_dir,
            slide_images=slide_images,
            vision_model=vision_model,
            use_cache=False,
        )
        
        content_informativeness_score, content_informativeness_feedback = eval_content_informativeness(
            target_ppt_dir=target_ppt_dir,
            target_doc_path=target_doc_path,
            vision_model=vision_model,
            language_model=language_model,
            marker_model=marker_model,
            slide_images=slide_images,
            eval_dir=eval_target_id_dir,
            use_cache=False,
        )


        # coherence_score, coherence_feedback = eval_coherence(language_model=language_model, presentation_outline=target_pres_outline)

        results["scores"]["coverage_iou"] = automatic_similarity_score['iou']
        results["scores"]["flow_ngld"] = automatic_similarity_score['ngld']
        results["scores"]["content_structure_similarity"] = content_structure_similarity_score
        results["scores"]["aesthetic"] = aesthetic_quality_score
        results["scores"]["content"] = content_informativeness_score
        # results["scores"]["coherence"] = coherence_score
        results["scores"]["template_layout_similarity"] = template_similarity_score

        
        # Add detailed feedbacks
        # structured_coherence_feedback = {
        #     "reason": coherence_feedback,
        #     "score": coherence_score
        # }
        
        structured_content_structure_feedback = {
            "reason": content_structure_similarity_feedback,
            "score": content_structure_similarity_score
        }
        
        structured_aesthetic_feedback = {
            "reason": aesthetic_quality_feedback,
            "score": aesthetic_quality_score
        }
        
        structured_content_informativeness_feedback = {
            "reason": content_informativeness_feedback,
            "score": content_informativeness_score
        }
        
        structured_template_similarity_feedback = {
            "reason": template_similarity_feedback,
            "score": template_similarity_score
        }
        
        
        results["aesthetic_feedback"] = structured_aesthetic_feedback
        results["content_informativeness_feedback"] = structured_content_informativeness_feedback
        # results["coherence_feedback"] = structured_coherence_feedback
        results["template_similarity_feedback"] = structured_template_similarity_feedback
        results["content_structure_similarity_feedback"] = structured_content_structure_feedback

        # Save evaluation results
        eval_path = os.path.join(eval_base_dir, f"{target_id}_evaluation.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


    except Exception as e:
        print(f"[ERROR] Error evaluating presentation: {e}")
        import traceback
        traceback.print_exc()
        # import pdb; pdb.set_trace()
        # return None
    
    return results


def scale_average_scores(input_file, output_file):
    """
    Loads evaluation data from a JSON file, scales the average scores
    according to predefined rules, and prints the results.
    """
    # --- Configuration ---
    # The path to your JSON file containing the scores.

    print(f"--- Scaling Scores from: {input_file} ---")
    
    # --- 1. Load the JSON data ---
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'. Please check the file format.")
        return
    except KeyError:
        print(f"Error: 'avg_scores' key not found in '{input_file}'.")
        return

    # --- 2. Get the original scores ---
    original_scores = data.get('avg_scores')
    if not original_scores:
        print("Error: 'avg_scores' dictionary is missing or empty in the JSON file.")
        return

    print("\nOriginal Scores:")
    for key, value in original_scores.items():
        print(f"  {key}: {value:.4f}")

    # --- 3. Scale the scores ---
    scaled_scores = {}
    scaling_factors = {
        'coverage_iou': 100,
        'flow_ngld': 100
    }
    default_scaling_factor = 20

    for key, value in original_scores.items():
        factor = scaling_factors.get(key, default_scaling_factor)
        scaled_scores[key] = value * factor

    # --- 4. Print the scaled scores ---
    print("\nScaled Scores:")
    for key, value in scaled_scores.items():
        print(f"  {key}: {value:.2f}")
    
    # --- 5. Save the scaled scores to a new JSON file ---
    with open(output_file, 'w') as f:
        json.dump(scaled_scores, f, indent=4)
    
    print("\n--- Script Finished ---")


def _evaluate(args):
    
    eval_task_name = os.path.basename(args.eval_result_path).replace('.json', '')
    
    with open('api_key.json', 'r') as f:
        api_key = json.load(f)
        openai_api_key = api_key['openai_api_key']
        
    with open(args.gen_config, 'r') as f:
        gen_config = json.load(f)

    # Setup models
    
    if args.local_llm and args.llm_api_server_url and args.vlm_api_server_url:
            # Use OpenAI client with local API server URL
            print(f"[INFO] Using OpenAI-compatible API server at: {args.llm_api_server_url}")
            
            # Setup online models with custom API base URL
            llm_client = OpenAI(api_key='YOUR_API_KEY', base_url=f"{args.llm_api_server_url}/v1")
            model_name = llm_client.models.list().data[0].id
            language_model = LLM(model=model_name, api_base=f"{args.llm_api_server_url}/v1", api_key="dummy-key")
            
            vlm_client = OpenAI(api_key='YOUR_API_KEY', base_url=f"{args.vlm_api_server_url}/v1")
            model_name = vlm_client.models.list().data[0].id
            vision_model = LLM(model=model_name, api_base=f"{args.vlm_api_server_url}/v1", api_key="dummy-key")
    else:
        language_model = LLM(model="gpt-4.1-2025-04-14", api_key=openai_api_key)
        vision_model = LLM(model="gpt-4.1-2025-04-14", api_key=openai_api_key)
        
    marker_model = create_model_dict(device=args.device, dtype=torch.float16)
    setup_models(language_model, vision_model)
    
    # set narrative categories (templated sections)
    narrative_categories = ["Title and Authors", "Definitions and Background", "Motivation and Challenges", "Related Work", "Dataset Construction", "Problem Formulation", "Method Overview", "Experiment Setup and Results", "Future Directions", "Summary and Conclusions"]

    # Load existing scores if the result file exists
    if os.path.exists(args.eval_result_path):
        print(f"[INFO] Found existing results file at {args.eval_result_path}. Loading scores.")
        with open(args.eval_result_path, 'r') as f:
            try:
                scores = json.load(f)
            except json.JSONDecodeError:
                print("[WARNING] Could not decode JSON from existing results file. Starting fresh.")
                scores = {}
    else:
        scores = {
            "avg_scores": {},
        }
        
    for conf in tqdm(gen_config):
        target_fname = conf['target']
        print(f"evaluating {target_fname}")
        
        target_id = os.path.splitext(target_fname)[0]

        if target_id in scores and scores.get(target_id) is not None:
            print(f"[INFO] Skipping {target_id} as it is already evaluated.")
            # continue

        else:    
            # target_ppt_path = os.path.join(args.target_ppt_dir, target_id, 'final.pptx')
            target_ppt_path = os.path.join(args.target_ppt_dir, f'{target_id}.pptx')
            # target_doc_path = os.path.join(args.sample_dir, "paper", target_fname)
            target_doc_path = os.path.join(args.target_paper_dir, f'{target_id}.pdf')
            
            template_fname = conf['template']
            template_ppt_path = os.path.join(args.template_ppt_dir, template_fname)
            
            sample_fname = conf['sample']['ppt']
            sample_id = os.path.splitext(sample_fname)[0]
            
                    
            results = evaluate_presentation(eval_task_name=eval_task_name,
                                            vision_model=vision_model,
                                            language_model=language_model,
                                            marker_model=marker_model,
                                            target_id=target_id,
                                            target_doc_path=target_doc_path,
                                            target_ppt_path=target_ppt_path,
                                            sample_id=sample_id,
                                            sample_dir=args.sample_dir,
                                            template_ppt_path=template_ppt_path,
                                            narrative_categories=narrative_categories)
            
            scores[target_id] = results
        
        # Recalculate average scores and save results after each evaluation
        try:
            avg_scores = {}
            scores_for_avg = {k: v for k, v in scores.items() if k != "avg_scores" and v is not None}

            if scores_for_avg:
                for target_result in scores_for_avg.values():
                    if 'scores' in target_result:
                        for k, v in target_result['scores'].items():
                            if k not in avg_scores:
                                avg_scores[k] = 0
                            avg_scores[k] += v
                
                num_valid_scores = len(scores_for_avg)
                if num_valid_scores > 0:
                    for k in avg_scores:
                        avg_scores[k] = avg_scores[k] / num_valid_scores
            
            scores["avg_scores"] = avg_scores
            
            with open(args.eval_result_path, 'w') as f:
                json.dump(scores, f, indent=4)

        except Exception as e:
            print(f"[ERROR] Error calculating average scores and saving: {e}")
            # Save partial results without averages if calculation fails
            with open(args.eval_result_path.replace('.json', '_partial_no_avg.json'), 'w') as f:
                json.dump(scores, f, indent=4)
            import traceback
            traceback.print_exc()
    
            
    final_avg_scores = scores.get("avg_scores", {})
    print(f"Evaluation average scores: {final_avg_scores}")
    print(f"Evaluation results saved to {args.eval_result_path}.")
    
    analysis_path = args.eval_result_path.replace('.json', '_scaled.json')
    scale_average_scores(args.eval_result_path, analysis_path)


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--target_ppt_dir', type=str, default="runs",
                        help='Directory containing the pptxs to be evaluated')
    parser.add_argument('--target_paper_dir', type=str, default="doc2slide_dataset/target_papers",
                        help='Directory containing the pptxs to be evaluated')
    parser.add_argument('--sample_dir', type=str, default="doc2slide_dataset/slide_paper_pairs",
                        help='Directory containing the samples')
    parser.add_argument('--template_ppt_dir', type=str, default="doc2slide_dataset/slide_templates",
                        help='Directory containing the template pptxs')
    parser.add_argument('--gen_config', type=str, default="example_runs/gen_config_eval_test.json",
                    help='Generation config used for the generation')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:X or mps')
    parser.add_argument('--eval_result_path', type=str, default="example_runs/eval_v2_sample2_scores.json",
                    help='file to store the results')
    parser.add_argument('--local_llm', action='store_true', default=False,
                        help='Use local LLM')
    parser.add_argument('--vlm_api_server_url', type=str, default="http://0.0.0.0:8001",
                        help='URL of OpenAI-compatible VLM API server (e.g., vllm or lmdeploy, http://0.0.0.0:8001)')
    parser.add_argument('--llm_api_server_url', type=str, default="http://0.0.0.0:8002",
                        help='URL of OpenAI-compatible LLM API server (e.g., vllm or lmdeploy, http://0.0.0.0:8002)')

    args = parser.parse_args()
    
    _evaluate(args)