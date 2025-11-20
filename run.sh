#！/bin/bash

python run_pptgen_refine_config.py \
--config_file doc2slide_dataset/configs/example_config.json \
--dataset_dir doc2slide_dataset \
--slides 10 \
--device "cuda:0" \
--no_refinement 
