from google import genai
import os

def enhance_prompt(image, prompt):
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    
    # input_caption_prompt = "Please provide a prompt for the image for Diffusion Model text-to-image generative model training, i.e. for FLUX or StableDiffusion 3. The prompt should be a detailed description of the image, including the character/asset/item, the environment, the pose, the lighting, the camera view, etc. The prompt should be detailed enough to generate the image. The prompt should be as short and precise as possible, in one-line format, and does not exceed 77 tokens."
    input_caption_prompt = (
        "Please provide a prompt for a Diffusion Model text-to-image generative model for the image I will give you. "
        "The prompt should be a detailed description of the image, especially the main subject (i.e. the main character/asset/item), the environment, the pose, the lighting, the camera view, the style etc."
        "The prompt should be detailed enough to generate the target image. "
        # "Identify key elements and that remain consistent from the source image, and highlight differences in the target image. "
        # "The prompt should be short, precise, one-line, similar to LAION dataset style, and not exceed 77 tokens. "
        # "The prompt should be detailed enough to generate the target image."
        "The prompt should be short and precise, in one-line format, and does not exceed 77 tokens."
        "The prompt should be individually coherent as a description of the image."
    )
    


    # Choose a Gemini model.
    caption_model = genai.Client(api_key=api_key)
    input_image_prompt = caption_model.models.generate_content(
        model='gemini-1.5-flash', contents=[input_caption_prompt, image]).text
    input_image_prompt = input_image_prompt.replace('\r', '').replace('\n', '')

    enhance_instruction = "Enhance this input text prompt: '"
    enhance_instruction += prompt
    enhance_instruction += "'. Please extract other details, especially description of the main subject from the following reference prompt: '"
    enhance_instruction += input_image_prompt
    enhance_instruction += "'. Please keep the details that are mentioned in the input prompt, and enhance the rest. "
    enhance_instruction += "Response with only the enhanced prompt. "
    enhance_instruction += "The enhanced prompt should be short and precise, in one-line format, and does not exceed 77 tokens."
    enhanced_prompt = caption_model.models.generate_content(
        model='gemini-1.5-flash', contents=[enhance_instruction]).text.replace('\r', '').replace('\n', '')
    print("input_image_prompt: ", input_image_prompt)
    print("prompt: ", prompt)
    print("enhanced_prompt: ", enhanced_prompt)
    return enhanced_prompt