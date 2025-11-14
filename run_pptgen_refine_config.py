#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from openai import OpenAI

# Add src directory to path
os.sys.path.append('./src')

from llms import LLM, setup_models
from utils import Config, pjoin, pptx_to_pdf
from agentic_loop.pref_refine_loop import refine_loop, refine_loop_with_cache

# Constants
RUNS_DIR = "runs"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate presentations from config file with reference and refinement')
    
    parser.add_argument('--config_file', type=str, required=True,
                    help='Path to config JSON file containing generation tasks')
    parser.add_argument('--output_dir', type=str,
                    help='Base output directory for generated presentations')
    parser.add_argument('--slides', type=int, default=5,
                    help='Number of slides to generate')
    parser.add_argument('--device', type=str, default='cuda:0', 
                    help='Device to run models on')

    # dataset specific arguments
    parser.add_argument('--dataset_dir', type=str, default="doc2slide_dataset",
                        help='Path to dataset directory')
    

    # agentic loop specific arguments
    parser.add_argument('--iterations', type=int, default=3,
                        help='Maximum number of refinement iterations')
    parser.add_argument('--no_refinement', action='store_true',
                        help='Disable refinement loop')
    parser.add_argument('--use_cache', type=bool, default=True,
                        help='Use cached results')
    parser.add_argument('--regen_outline', action='store_true',
                        help='Regenerate outline')
    
    # LLM 
    parser.add_argument('--local_llm', action='store_true',
                        help='Use local LLM instead of OpenAI API')
    parser.add_argument('--local_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help='Path to local model when using --local_llm')
    parser.add_argument('--vlm_api_server_url', type=str, default="http://0.0.0.0:8001",
                        help='URL of OpenAI-compatible VLM API server (e.g., vllm or lmdeploy, http://0.0.0.0:8001)')
    parser.add_argument('--llm_api_server_url', type=str, default="http://0.0.0.0:8002",
                        help='URL of OpenAI-compatible LLM API server (e.g., vllm or lmdeploy, http://0.0.0.0:8002)')
    return parser.parse_args()


def setup_models_from_args(args):
    if args.local_llm:
        if args.llm_api_server_url and args.vlm_api_server_url:
            # Use OpenAI client with local API server URL
            print(f"[INFO] Using OpenAI-compatible API server at: {args.llm_api_server_url}")
            
            # Setup online models with custom API base URL
            llm_client = OpenAI(api_key='YOUR_API_KEY', base_url=f"{args.llm_api_server_url}/v1")
            model_name = llm_client.models.list().data[0].id
            language_model = LLM(model=model_name, api_base=f"{args.llm_api_server_url}/v1", api_key="dummy-key")
            
            vlm_client = OpenAI(api_key='YOUR_API_KEY', base_url=f"{args.vlm_api_server_url}/v1")
            model_name = vlm_client.models.list().data[0].id
            vision_model = LLM(model=model_name, api_base=f"{args.vlm_api_server_url}/v1", api_key="dummy-key")
            
            # vision_model = language_model  # Share the same model instance
        else:
            # deploy a local model
            print(f"[INFO] Using local LLM: {args.local_model_path}")
            from lmdeploy import VisionConfig, GenerationConfig
            # Configure vision and generation parameters for local model
            vision_config = VisionConfig(image_size=1024)
            gen_config = GenerationConfig(max_new_tokens=4096)
            
            # Use the same model for both language and vision
            language_model = LLM(model=args.local_model_path, offline_inference=True, 
                                vision_config=vision_config, gen_config=gen_config)
            vision_model = language_model  # Share the same model instance
    else:
        # Load API key for OpenAI
        with open('api_key.json', 'r') as f:
            api_key = json.load(f)
            openai_api_key = api_key['openai_api_key']
        
        openai_model = "gpt-4.1-2025-04-14" # "gpt-4.1-2025-04-14"
        
        # Setup online models
        language_model = LLM(model=openai_model, api_key=openai_api_key)
        vision_model = LLM(model=openai_model, api_key=openai_api_key)
    
    setup_models(language_model, vision_model)
    return language_model, vision_model

def process_config_item(config_item, args, base_output_dir):
    target_id = os.path.splitext(config_item['target'])[0]
    ref_id = os.path.splitext(config_item['sample']['paper'])[0]

    ppt_template = os.path.join(args.dataset_dir, "slide_templates", config_item['template'])
    target_pdf = os.path.join(args.dataset_dir, "target_papers", f"{target_id}.pdf")
    ref_content_pdf = os.path.join(args.dataset_dir, "slide_paper_pairs", "paper", f"{ref_id}.pdf")
    ref_content_ppt = os.path.join(args.dataset_dir, "slide_paper_pairs", "ppt", f"{ref_id}.pdf")
    
    # Create unique output directory for this task
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:4]
    project_id = f"{dt_str}_{unique_id}_ref{ref_id}_tgt{target_id}"
    output_dir = os.path.join(base_output_dir, project_id)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Processing task: Target={target_id}, Reference={ref_id}")
    print(f"[INFO] Using run directory: {output_dir}")
    
    # Load models
    text_model = None
    image_model = None
    marker_model = None
    
    # Setup config
    generation_config = Config(output_dir)
    pptx_md5 = hashlib.md5(open(ppt_template, "rb").read()).hexdigest()
    pptx_config = Config(pjoin(output_dir, "pptx", pptx_md5))
    os.makedirs(pptx_config.RUN_DIR, exist_ok=True)
    
    # Create a modified args object with the current parameters
    from types import SimpleNamespace
    task_args = SimpleNamespace(
        slides=args.slides,
        iterations=args.iterations,
        no_refinement=args.no_refinement,
        regen_outline=args.regen_outline,
        device=args.device,
        target_id=target_id,
        ref_id=ref_id
    )
    
    # Run refinement loop
    final_pptx_path = refine_loop_with_cache(
        ppt_path=ppt_template,
        ref_content_pdf=ref_content_pdf,
        ref_content_ppt=ref_content_ppt,
        target_pdf=target_pdf,
        args=task_args,
        project_id=project_id,
        output_dir=output_dir,
        generation_config=generation_config,
        pptx_config=pptx_config,
        vision_model=vision_model,
        language_model=language_model,
        text_model=text_model,
        image_model=image_model,
        marker_model=marker_model,
        use_cache=args.use_cache,
    )
    
    if final_pptx_path:
        print(f"[SUCCESS] Final presentation: {final_pptx_path}")
        
        # Convert to PDF for easier viewing
        pptx_to_pdf(final_pptx_path, os.path.dirname(final_pptx_path))
        print(f"[INFO] PDF version: {final_pptx_path.replace('.pptx', '.pdf')}")
        return True
    else:
        print(f"[ERROR] Presentation generation failed for task: Target={target_id}, Reference={ref_id}")
        return False

def main():
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"[ERROR] Config file not found: {args.config_file}")
        sys.exit(1)
    
    # Load config file
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
    
    # Create base output directory
    if not args.output_dir:
        base_output_dir = RUNS_DIR
    else:
        base_output_dir = args.output_dir
    
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"[INFO] Base output directory: {base_output_dir}")
    
    # Setup models once for all tasks
    try:
        language_model, vision_model = setup_models_from_args(args)
        
        # Save models for the global scope to be passed to each task
        globals()['language_model'] = language_model
        globals()['vision_model'] = vision_model
    except Exception as e:
        print(f"[ERROR] Failed to setup models: {str(e)}")
        sys.exit(1)
    
    # Process each config item
    results = []
    for i, config_item in enumerate(config_data):
        print(f"\n[INFO] Processing task {i+1}/{len(config_data)}")
        print(f"[INFO] Task item: {config_item}")
        try:
            success = process_config_item(config_item, args, base_output_dir)
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"[ERROR] Task {i+1} failed: {error_msg}")
        
        results.append({
            "task": i+1,
            "target": config_item['target'],
            "reference": config_item['sample']['paper'],
            "success": success,
            "error": error_msg
        })
    # Save summary of results
    summary_path = os.path.join(base_output_dir, f"summary_{args.config_file.split('/')[-1].split('.')[0]}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\n[SUMMARY] Completed {success_count}/{len(results)} tasks successfully")
    print(f"[SUMMARY] Results saved to: {summary_path}")
    
    # Exit with error if any task failed
    if success_count < len(results):
        sys.exit(1)

if __name__ == "__main__":
    main() 