#！/bin/bash

python run_pptgen_refine_config.py \
--config_file doc2slide_dataset/config_0724_target150_template10/slidegen_config_part1.json \
--slides 10 \
--device "cuda:0" \
--no_refinement 
