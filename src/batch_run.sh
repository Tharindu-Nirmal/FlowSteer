#!/bin/bash

# Runs one demo image through all four FlowSteer restoration tasks.
# Before running this script:
#   1. Add your images to demo/inputs/
#   2. Run: python demo/prepare_demo.py   (creates degraded versions in demo/degraded/)
#   3. Run: cd src && bash batch_run.sh

PYTHON_SCRIPT="edit_image.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"
DEMO_DIR="../demo"
OUTPUT_BASE="$DEMO_DIR/outputs"

# Edit the --source_img_dir paths and prompts to match your image content.
runs=(
# --- sample_cat ---
"--source_prompt \"A low resolution image of a cat. There are blocking artifacts on the image.\" --target_prompt \"A high resolution image of a cat. Sharp fur details, photorealistic. No blocking artifacts.\" --degradation \"super resolution\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.70 --lambda_end 0.85 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 3 --source_img_dir $DEMO_DIR/degraded/superres_4x/sample_cat.jpg"
"--source_prompt \"A black and white image of a cat.\" --target_prompt \"A full color image of a cat. Natural, realistic fur colors.\" --degradation \"colorization\" --guidance 4 --inject 5 --lambda_start 0.40 --lambda_step 0.50 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.8 --lambda_final_pad 1 --source_img_dir $DEMO_DIR/degraded/colorized/sample_cat.jpg"
"--source_prompt \"A noisy image of a cat.\" --target_prompt \"A clean, noise-free image of a cat. Sharp fur details, photorealistic. No noise artifacts.\" --degradation \"denoising\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.75 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 2 --source_img_dir $DEMO_DIR/degraded/denoised/sample_cat.jpg"
"--source_prompt \"A blurry image of a cat.\" --target_prompt \"A sharp image of a cat. Clear fur details, photorealistic.\" --degradation \"deblurring\" --guidance 4 --inject 5 --lambda_start 0.70 --lambda_step 0.80 --lambda_end 0.90 --lambda_level_hi 1.0 --lambda_level_lo 0.3 --lambda_final_pad 3 --source_img_dir $DEMO_DIR/degraded/deblurred/sample_cat.jpg"
# --- sample_portrait ---
"--source_prompt \"A low resolution portrait photograph. There are blocking artifacts on the image.\" --target_prompt \"A high resolution portrait photograph. Sharp facial details, photorealistic. No blocking artifacts.\" --degradation \"super resolution\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.70 --lambda_end 0.85 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 3 --source_img_dir $DEMO_DIR/degraded/superres_4x/sample_portrait.jpg"
"--source_prompt \"A black and white portrait photograph.\" --target_prompt \"A full color portrait photograph. Natural, realistic skin tones.\" --degradation \"colorization\" --guidance 4 --inject 5 --lambda_start 0.40 --lambda_step 0.50 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.8 --lambda_final_pad 1 --source_img_dir $DEMO_DIR/degraded/colorized/sample_portrait.jpg"
"--source_prompt \"A noisy portrait photograph.\" --target_prompt \"A clean, noise-free portrait photograph. Sharp facial details, photorealistic. No noise artifacts.\" --degradation \"denoising\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.75 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 2 --source_img_dir $DEMO_DIR/degraded/denoised/sample_portrait.jpg"
"--source_prompt \"A blurry portrait photograph.\" --target_prompt \"A sharp, in-focus portrait photograph. Clear facial details, photorealistic.\" --degradation \"deblurring\" --guidance 4 --inject 5 --lambda_start 0.70 --lambda_step 0.80 --lambda_end 0.90 --lambda_level_hi 1.0 --lambda_level_lo 0.3 --lambda_final_pad 3 --source_img_dir $DEMO_DIR/degraded/deblurred/sample_portrait.jpg"
)

mkdir -p "$OUTPUT_BASE"

for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1)) of ${#runs[@]}..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done

echo "Done. Results saved to $OUTPUT_BASE"
