import argparse
import torch
from PIL import Image
from diffusers.utils import load_image

from transformers import AutoModelForCausalLM

import sys
sys.path.append(".")

from pipeline import FluxConditionalPipeline
from transformer import FluxTransformer2DConditionalModel


def process_image_and_text(
    pipe, image, text, gemini_prompt, guidance, i_guidance, t_guidance, steps, height, width
):
    """Process the given image and text using the global pipeline."""
    
    image = resize_and_center_crop(image, height, width//2)

    control_image = load_image(image)

    if gemini_prompt:
        from recaption import enhance_prompt
        text = enhance_prompt(image, text.strip().replace("\n", "").replace("\r", ""))

    result = pipe(
        prompt=text.strip().replace("\n", "").replace("\r", ""),
        negative_prompt="",
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance_scale=guidance,
        image=control_image,
        guidance_scale_real_i=i_guidance,
        guidance_scale_real_t=t_guidance,
    ).images[0]

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diffusion Self-Distillation.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/pytorch_lora_weights.safetensors",
        help="Path to the lora checkpoint.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument("--text", type=str, required=True, help="The text prompt.")
    parser.add_argument(
        "--disable_gemini_prompt",
        action="store_true",
        help="Flag to disable gemini prompt. If not set, gemini_prompt is True.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3.5, help="Guidance scale for the pipeline."
    )
    parser.add_argument(
        "--i_guidance", type=float, default=1.0, help="Image guidance scale."
    )
    parser.add_argument(
        "--t_guidance", type=float, default=1.0, help="Text guidance scale."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image.",
    )
    #height and width
    parser.add_argument(
        "--height", type=int, default=512, help="Height of the output image."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the output image."
    )
    parser.add_argument(
        "--sequential_offload",
        action="store_true",
        help="Sequentially offload to CPU",
    )
    parser.add_argument(
        "--model_offload", action="store_true", help="Offload full models"
    )
    parser.add_argument("--steps", type=int, default=28, help="Steps to generate")
    

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    print(f"LoRA: {args.lora_path}")

    # Initialize pipeline
    transformer = FluxTransformer2DConditionalModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    )
    pipe = FluxConditionalPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    
    pipe.scheduler.config.shift = 3
    pipe.scheduler.config.use_dynamic_shifting = True
    
    assert isinstance(pipe, FluxConditionalPipeline)
    pipe.load_lora_weights(args.lora_path)
    if args.model_offload:
        pipe.enable_model_cpu_offload()
    if args.sequential_offload:
        pipe.enable_sequential_cpu_offload()
    if not args.model_offload and not args.sequential_offload:
        pipe.to("cuda")

    # Open the image
    image = Image.open(args.image_path).convert("RGB")

    print(f"Process image: {args.image_path}")

    # Process image and text
    result_image = process_image_and_text(
        pipe,
        image,
        args.text,
        not args.disable_gemini_prompt,
        args.guidance,
        args.i_guidance,
        args.t_guidance,
        args.steps,
        args.height,
        args.width
    )

    # Save the output
    result_image.save(args.output_path)
    print(f"Output saved to {args.output_path}")


def resize_and_center_crop(image: Image.Image, target_height: int = 512, target_width: int = 512) -> Image.Image:

    
    # Handle PIL Image
    w, h = image.size
    min_size = min(w, h)

    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    # Calculate current aspect ratio
    current_ratio = w / h

    # Resize to match target width or height while preserving aspect ratio
    if current_ratio > target_ratio:
        # Image is wider than target - resize by height
        new_height = target_height
        new_width = int(w * (target_height / h))
    else:
        # Image is taller than target - resize by width
        new_width = target_width
        new_height = int(h * (target_width / w))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Center crop the image to the target size
    cropped = image.crop(((new_width - target_width) // 2, 
                         (new_height - target_height) // 2, 
                         (new_width + target_width) // 2, 
                         (new_height + target_height) // 2))

    return cropped

if __name__ == "__main__":
    main()
    # C:\Users\emreb\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe generate.py --model_path C:\Users\emreb\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\dsd_model\transformer --lora_path C:\Users\emreb\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\dsd_model\pytorch_lora_weights.safetensors --image_path C:\Users\emreb\Downloads\image.webp --text "this character sitting on a chair" --output_path output.png
