import os
from glob import glob
from os.path import join as pjoin

from tqdm import tqdm

from src.utils import pptx_to_pdf

if __name__ == "__main__":
    input_dir = "results_finalized/pptx_0726_chatgpt_final"
    output_dir = pjoin("results_finalized", "pdf_0726_chatgpt_final")
    os.makedirs(output_dir, exist_ok=True)

    files_to_convert = glob(pjoin(input_dir, "*.pptx"))
    for f in tqdm(files_to_convert):
        try:
            pptx_to_pdf(f, output_dir)
        except Exception as e:
            print(f"Failed to convert {f}: {e}")
    print(f"Successfully converted {len(files_to_convert)} files.") 