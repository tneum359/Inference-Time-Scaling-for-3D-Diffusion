import torch
from diffusers import StableZero123Pipeline
from PIL import Image
import os
import json
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import argparse
import random
import requests
from io import BytesIO
import base64
import typing

from verifiers.gemini_verifier import GeminiVerifier # Added
from laion_aesthetics import LAIONAestheticVerifier

# --- Helper Functions (Integrated from utils.py) ---
def load_image(image_path_or_url):
    """Loads an image from a path or URL."""
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    return image

def convert_to_bytes(image: Image.Image, b64_encode=False) -> typing.Union[bytes, str]:
    """Converts a PIL Image to bytes (PNG format for Gemini)."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    # Base64 encoding is not needed for Gemini bytes input unless specifically required elsewhere
    if b64_encode:
        return base64.b64encode(img_bytes).decode("utf-8")
    return img_bytes

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and verify multiview images using StableZero123 and Gemini/LAION.")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidate multiview image groups to generate and evaluate."
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        help="Path or URL to the input image for StableZero123."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Photo of an athlete cat explaining its latest scandal at a press conference to journalists.",
        help="Prompt to use for the Gemini verifier."
    )
    return parser.parse_args()

# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()

    # --- Configuration ---
    # Load environment variables (for GEMINI_API_KEY)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set. Please create a .env file.")
        exit(1)

    # Model and Pipeline
    model_id = "stabilityai/stable-zero123"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Input Data from args
    input_image_path_or_url = args.input_image
    prompt_text = args.prompt
    num_candidates = args.num_candidates

    choice_of_metric_val = "average_overall_score" # Metric for selecting the best *group*

    # Output Configuration (Modified Structure)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct path closer to the example format
    run_output_dir = os.path.join("output", "stable-zero123", "gemini", "overall_score", timestamp)
    # Create the base output directory if it doesn't exist
    os.makedirs(os.path.dirname(run_output_dir), exist_ok=True)
    # Don't create the final timestamped directory yet

    # --- Load Model & Verifiers ---
    print(f"Loading model: {model_id}")
    try:
        pipeline = StableZero123Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipeline.to(device)
        print("Successfully imported StableZero123Pipeline!")
    except ImportError as e:
        print(f"Import failed: {e}")
        import diffusers
        print(f"Diffusers location: {diffusers.__file__}")
        print(f"Diffusers version: {diffusers.__version__}")
        pipeline = diffusers.StableZero123Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipeline.to(device)

    print("Initializing verifiers...")
    gemini_verifier = GeminiVerifier() # Requires GEMINI_API_KEY in .env
    laion_verifier = LAIONAestheticVerifier(dtype=dtype)

    # --- Load Input Image ---
    print(f"Loading input image: {input_image_path_or_url}")
    try:
        input_image = load_image(input_image_path_or_url)
    except Exception as e:
        print(f"Error loading input image: {e}")
        exit(1)

    # --- Generation and Verification Loop ---
    best_group_data = {
        "avg_score": -1,
        "images": None,
        "gemini_scores": None,
        "laion_scores": None,
        "seed": -1
    }

    print(f"Generating and evaluating {num_candidates} candidate groups...")
    for i in range(num_candidates):
        print(f"\n--- Candidate Group {i+1}/{num_candidates} ---")
        # Use a unique seed for each candidate group generation
        current_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(current_seed)

        # 1. Generate Multiview Images for this group
        print(f"Generating multiview images with seed: {current_seed}...")
        try:
            generated_images = pipeline(
                input_image,
                num_inference_steps=28,
                generator=generator
            ).images
            print(f"Generated {len(generated_images)} views.")
        except Exception as e:
            print(f"Error during image generation for group {i+1}: {e}")
            continue # Skip to the next candidate group

        # 2. Apply Gemini Verifier
        print("Applying Gemini Verifier...")
        gemini_prompts = [prompt_text] * len(generated_images)
        try:
            gemini_inputs = gemini_verifier.prepare_inputs(images=generated_images, prompts=gemini_prompts)
            gemini_results = gemini_verifier.score(inputs=gemini_inputs) # List of dicts, one per image
            # Calculate average overall score for the group
            group_overall_scores = [res.get("overall_score", {}).get("score", 0) for res in gemini_results]
            avg_group_score = np.mean(group_overall_scores) if group_overall_scores else 0
            print(f"Gemini Average Overall Score for Group: {avg_group_score:.4f}")
        except Exception as e:
            print(f"Error during Gemini verification for group {i+1}: {e}")
            avg_group_score = -1 # Penalize groups that fail verification
            gemini_results = [] # Ensure it's an empty list

        # 3. Apply LAION Aesthetic Verifier (Optional, but kept for info)
        print("Applying LAION Aesthetic Verifier...")
        try:
            laion_inputs = laion_verifier.prepare_inputs(images=generated_images)
            laion_results = laion_verifier.score(inputs=laion_inputs)
            avg_laion_score = np.mean([res["laion_aesthetic_score"] for res in laion_results])
            print(f"LAION Average Aesthetic Score for Group: {avg_laion_score:.4f}")
        except Exception as e:
            print(f"Error during LAION verification for group {i+1}: {e}")
            laion_results = [] # Ensure it's an empty list

        # 4. Check if this group is the best so far
        if avg_group_score > best_group_data["avg_score"]:
            print(f"New best group found with average Gemini score: {avg_group_score:.4f}")
            best_group_data["avg_score"] = avg_group_score
            best_group_data["images"] = generated_images
            best_group_data["gemini_scores"] = gemini_results
            best_group_data["laion_scores"] = laion_results
            best_group_data["seed"] = current_seed

    # --- Process and Save Best Group --- 
    if best_group_data["images"] is None:
        print("\nNo successful candidate groups were generated or verified.")
        exit(1)

    print(f"\n--- Best Group (Seed: {best_group_data['seed']}) --- ")
    print(f"Average Gemini Overall Score: {best_group_data['avg_score']:.4f}")

    # Create the specific run directory for the best group now
    os.makedirs(run_output_dir, exist_ok=True)

    # Save images from the best group
    saved_image_paths = []
    best_gemini_scores = best_group_data["gemini_scores"]
    for i, img in enumerate(best_group_data["images"]):
        prompt_slug = "".join(c if c.isalnum() else '_' for c in prompt_text[:50])
        img_score = best_gemini_scores[i].get('overall_score', {}).get('score', 'N/A') if best_gemini_scores else 'N/A'
        img_score_str = f"{img_score:.1f}" if isinstance(img_score, (int, float)) else str(img_score)
        # Adjusted filename: using seed (s) and view index (i)
        img_filename = f"prompt@{prompt_slug}_s@{best_group_data['seed']}_i@{i}_score@{img_score_str}.png"
        img_path = os.path.join(run_output_dir, img_filename)
        try:
            img.save(img_path)
            relative_img_path = os.path.relpath(img_path, start=os.getcwd())
            saved_image_paths.append(relative_img_path)
        except Exception as e:
            print(f"Error saving image {img_filename}: {e}")

    print(f"Saved {len(saved_image_paths)} images from the best group to: {run_output_dir}")

    # Calculate average LAION score for the best group (if available)
    best_laion_scores = best_group_data["laion_scores"]
    avg_laion_score_best_group = np.mean([res["laion_aesthetic_score"] for res in best_laion_scores]) if best_laion_scores else None

    # --- Prepare Final Output (Using Group Average Score) --- 
    output_data = {
        "prompt": prompt_text,
        "best_score": {
            "explanation": "Average Gemini 'overall_score' across all views in the best group.",
            "score": best_group_data["avg_score"] # Use the average score of the best group
        },
        # Metric used to determine the *best group*
        "choice_of_metric": choice_of_metric_val, # Should be 'average_overall_score'
        # Relative path to the directory containing all images of the best group
        "best_group_dir": os.path.relpath(run_output_dir, start=os.getcwd())
    }

    # --- Save Results --- 
    results_filename = os.path.join(run_output_dir, "results.json")
    try:
        with open(results_filename, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results JSON saved to: {results_filename}")
    except Exception as e:
        print(f"Error saving results JSON: {e}")

    print("--- Script Finished ---")
