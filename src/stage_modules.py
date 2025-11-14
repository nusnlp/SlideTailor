#!/usr/bin/env python3
import json
import os
from copy import deepcopy
from typing import Dict


# Add src directory to path
os.sys.path.append('./src')

# Import project modules
# import induct
import induct_v2 as induct
# import pptgen
import pptgen_2stage as pptgen
from multimodal import ImageLabler
from presentation import Presentation
from utils import Config, pjoin, ppt_to_images, pptx_to_pdf
from doc_handling import generate_preference_presentation_guidelines, conditional_refine_document_with_guidelines
from pdf_parsing import parse_pdf, parsing_pdf_with_caption



def stage_ppt_template_parsing(ppt_path, pptx_config, vision_model):
    """
    Stage 1: PPT template parsing - Parse the PPT template and prepare images
    
    Args:
        ppt_path: Path to the PPT template
        pptx_config: Configuration object
        vision_model: Vision model for image captioning
        
    Returns:
        presentation: Parsed presentation object
    """
    print("[STAGE] PPT Parsing")
    presentation = Presentation.from_file(ppt_path, pptx_config)
    ppt_image_folder = pjoin(pptx_config.RUN_DIR, "slide_images")

    if not os.path.exists(ppt_image_folder) or len(os.listdir(ppt_image_folder)) == 0:
        ppt_to_images(ppt_path, ppt_image_folder)
        for err_idx, _ in presentation.error_history:
            err_path = pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg")
            if os.path.exists(err_path):
                os.remove(err_path)

        for i, slide in enumerate(presentation.slides, start=1):
            slide.slide_idx = i
            old_name = pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg")
            new_name = pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg")
            if os.path.exists(old_name):
                os.rename(old_name, new_name)

    labler = ImageLabler(vision_model=vision_model, presentation=presentation, config=pptx_config)
    labler.caption_images()
    
    return presentation, ppt_image_folder

def stage_slide_induction(presentation, ppt_image_folder, pptx_config, vision_model, language_model, image_model):
    """
    Stage 2: Slide induction (template analysis) - Analyze the template for slide structure
    
    Args:
        presentation: Parsed presentation object
        ppt_image_folder: Folder containing slide images
        pptx_config: Configuration object
        vision_model: Vision model
        language_model: Language model
        image_model: Image embedding model
        
    Returns:
        template_presentation: Template presentation object
        slide_induction: Slide induction data
    """
    print("[STAGE] Slide Induction")
    template_img_dir = pjoin(pptx_config.RUN_DIR, "template_images")
    if not os.path.exists(template_img_dir) or len(os.listdir(template_img_dir)) == 0:
        deepcopy(presentation).save(
            pjoin(pptx_config.RUN_DIR, "template.pptx"), layout_only=False
        )
        ppt_to_images(
            pjoin(pptx_config.RUN_DIR, "template.pptx"), template_img_dir
        )

    template_presentation = Presentation.from_file(
        pjoin(pptx_config.RUN_DIR, "template.pptx"), pptx_config
    )

    slide_inducter = induct.SlideInducter(
        vision_model,
        language_model,
        presentation,
        ppt_image_folder,
        template_img_dir,
        pptx_config,
        image_model,
        "inference_script",
    )
    slide_induction = slide_inducter.content_induct()
    print(f"Slide Induction:\n {os.path.join('./', slide_inducter.induct_cache)}")
    
    return template_presentation, slide_induction

def stage_reference_document_parsing(ref_content_pdf, ref_content_ppt, marker_model, vision_model, language_model, project_id, runs_dir = "runs"):
    """
    Stage 3: Reference document parsing - Parse reference PDF and PPT to extract presentation guidelines
    
    Args:
        ref_content_pdf: Path to reference PDF
        ref_content_ppt: Path to reference PPT
        marker_model: Model for PDF parsing
        vision_model: Vision model
        language_model: Language model
        project_id: Project identifier
        
    Returns:
        pref_guidelines: Presentation preference guidelines
    """
    print("[STAGE] PDF/Topic Parsing (Reference)")
    
    ref_pdf_parsed_pdf_dir = pjoin(runs_dir, project_id, "pdf", "ref_pdf")
    print(f"[INFO] Parsing reference PDF: {ref_content_pdf}")
    ref_pdf_md = parsing_pdf_with_caption(ref_content_pdf, ref_pdf_parsed_pdf_dir, marker_model, vision_model, language_model)
    # ref_pdf_md = parse_pdf(ref_content_pdf, ref_pdf_parsed_pdf_dir, marker_model)
    
    ref_slide_parsed_pdf_dir = pjoin(runs_dir, project_id, "pdf", "ref_slide_pdf")
    print(f"[INFO] Parsing reference PPT: {ref_content_ppt}")
    ref_slide_md = parsing_pdf_with_caption(ref_content_ppt, ref_slide_parsed_pdf_dir, marker_model, vision_model, language_model)
    # ref_slide_md = parse_pdf(ref_content_ppt, ref_slide_parsed_pdf_dir, marker_model)
    
    pref_guidelines = generate_preference_presentation_guidelines(language_model, ref_pdf_md, ref_slide_md)

    pref_guidelines_json_path = pjoin(runs_dir, project_id, "pref_guidelines.json")
    json.dump(pref_guidelines, open(pref_guidelines_json_path, "w"), indent=4)
    
    return pref_guidelines

def stage_target_document_parsing(pdf_path, marker_model, vision_model, language_model, project_id, pref_guidelines, runs_dir = "runs"):
    """
    Stage 4: Target document parsing - Parse target PDF with reference to guidelines
    
    Args:
        pdf_path: Path to target PDF
        marker_model: Model for PDF parsing
        vision_model: Vision model
        language_model: Language model
        project_id: Project identifier
        pref_guidelines: Presentation preference guidelines
        
    Returns:
        doc_json: Parsed document structure
        images: Images extracted from the document
    """
    print("[STAGE] PDF/Topic Parsing (Target)")
    
    parsed_pdf_dir = pjoin(runs_dir, project_id, "pdf", "target_pdf")
    text_content = parsing_pdf_with_caption(pdf_path, parsed_pdf_dir, marker_model, vision_model, language_model)
    
    parsedpdf_dir = pjoin(runs_dir, project_id, "pdf", "target_pdf")
    refined_doc_json_path = pjoin(parsedpdf_dir, "refined_doc.json")
    
    # Apply conditional refinement with guidelines
    doc_json = conditional_refine_document_with_guidelines(language_model, text_content, pref_guidelines)
    json.dump(doc_json, open(refined_doc_json_path, "w"), indent=4)

    # Load image captions if they exist
    caption_json_path = pjoin(parsedpdf_dir, "caption.json")
    images = json.load(open(caption_json_path)) if os.path.exists(caption_json_path) else {}
    
    return doc_json, images

def stage_presentation_generation(
    template_presentation, slide_induction, generation_config, 
    pref_guidelines, images, num_slides, doc_json,
    vision_model, language_model, text_model,
    presentation_outline = None
):
    """
    Stage 5: Presentation generation - Generate the final presentation
    
    Args:
        template_presentation: Template presentation object
        slide_induction: Slide induction data
        generation_config: Configuration for generation
        pref_guidelines: Presentation preference guidelines
        images: Images from the document
        num_slides: Number of slides to generate
        doc_json: Parsed document structure
        presentation_outline: Presentation outline
        vision_model: Vision model
        language_model: Language model
        text_model: Text embedding model
        
    Returns:
        output_pptx_path: Path to the generated presentation
    """
    
    # TODO: if given presentation outline, use it. else goes original generation process
    
    print("[STAGE] PPT Generation")
    crew = pptgen.PPTCrew(vision_model, language_model, text_model,
                        error_exit=True, retry_times=3)

    crew.set_reference(template_presentation,
                        slide_induction,
                        generation_config,
                        pref_guidelines
                        )

    output_pptx_path, presentation_outline = crew.generate_presentation(
        config=generation_config,
        images=images,
        num_slides=num_slides,
        doc_json=doc_json,
        presentation_outline=presentation_outline
    )
    
    if output_pptx_path is not None:    
        print("[STAGE] Success!")
        print(f"[INFO] Output PPT stored at {output_pptx_path}")
        
        # convert pptx to pdf
        pptx_to_pdf(output_pptx_path, os.path.dirname(output_pptx_path))
    else:
        print("[ERROR] Failed to generate PPT")
    
    return output_pptx_path, presentation_outline
