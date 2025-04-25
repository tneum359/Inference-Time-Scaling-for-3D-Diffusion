import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
import json
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import random
import requests
from io import BytesIO
import base64
import typing
import importlib.util
import pathlib
from huggingface_hub import snapshot_download
import re # Added for sanitizing filenames
import rembg # Added
from typing import Any # Added

from gemini_verifier import GeminiVerifier # Added
from laion_aesthetics import LAIONAestheticVerifier
from verifiers.camera_utils import get_circular_camera_poses # Added import

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

# --- Helper function to create a safe directory name --- Added
def get_input_name(path_or_url):
    try:
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            # Use the last part of the path, remove query params, remove extension
            name = os.path.splitext(path_or_url.split('?')[0].split('/')[-1])[0]
        else:
            # Use filename without extension
            name = os.path.splitext(os.path.basename(path_or_url))[0]
        # Sanitize
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        # Handle empty names
        return name if name else "default_input"
    except Exception:
        return "default_input" # Fallback

# --- Helper function for background removal --- Added
def remove_background(
    image: Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image.Image:
    do_remove = True
    # Skip if image is already transparent
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        print("  Input image already has transparency, skipping background removal.")
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        try:
            # Use the session for efficiency
            image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
        except Exception as e:
            print(f"  Error during rembg.remove: {e}")
            # Optionally re-raise or return original image based on desired handling
            # raise e # Or return the original image if failure is acceptable
            return image # Returning original image on failure for now
    return image

# --- Main Script --- Modified for Batch Processing
if __name__ == "__main__":
    # --- Static Configuration --- Define parameters here
    input_images = [
        # "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        "verifiers/mesh_origin.png", # Use local file
        # Add more image paths or URLs here, e.g.:
        # "path/to/your/local_image.png",
        # "https://another.domain/image.jpeg"
    ]
    num_candidates = 3
    prompt_text = "A high-quality 3D render of the object."
    # --- End Static Configuration ---

    # --- Configuration (Load Models/Verifiers ONCE outside the loop) ---
    # Load environment variables (for GEMINI_API_KEY)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set. Please create a .env file.")
        exit(1)

    # Model and Pipeline Setup
    model_id = "sudo-ai/zero123plus-v1.2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model: {model_id}")

    # --- Remove Dynamic Loading for Custom Pipeline ---
    # cache_dir = snapshot_download(model_id)
    # custom_pipeline_file = pathlib.Path(cache_dir) / "pipeline_zero123plus.py"
    # spec = importlib.util.spec_from_file_location("zero123plus_pipeline_module", custom_pipeline_file)
    # if spec is None or spec.loader is None:
    #     try:
    #         py_files = list(pathlib.Path(cache_dir).glob('*.py'))
    #         print(f"Error: Could not load spec. Found .py files: {py_files}")
    #     except Exception as list_e:
    #         print(f"Error: Could not load spec and also failed to list .py files in {cache_dir}: {list_e}")
    #     raise ImportError(f"Could not load spec for module from {custom_pipeline_file}")
    # zero123plus_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(zero123plus_module)
    # PipelineClass = zero123plus_module.Zero123PlusPipeline
    # --- End Dynamic Loading ---

    # Load pipeline using the recommended custom_pipeline argument
    # pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="sudo-ai/zero123plus-pipeline", # Specify the custom pipeline package
        torch_dtype=dtype
    )
    pipeline.to(device)

    # Verifier Setup
    print("Initializing verifiers...")
    gemini_verifier = GeminiVerifier() # Requires GEMINI_API_KEY in .env
    laion_verifier = LAIONAestheticVerifier(dtype=dtype)

    # Initialize rembg Session ONCE outside the loop
    print("Initializing background removal session...")
    try:
        rembg_session = rembg.new_session() # Use default model
        # You could specify a model like: rembg.new_session(model_name="u2net")
    except Exception as e:
        print(f"Error initializing rembg session: {e}")
        print("Background removal will likely fail.")
        rembg_session = None # Allow script to continue but removal will fail

    # --- Loop Over Input Images --- Added Outer Loop
    for input_image_path_or_url in input_images:
        print(f"\n{'='*20} Processing Input: {input_image_path_or_url} {'='*20}")

        # --- Per-Input Configuration ---
        input_name = get_input_name(input_image_path_or_url)
        choice_of_metric_val = "average_overall_score"

        # Simplified Output Directory Structure
        run_output_dir = os.path.join("output", input_name)
        # Create the specific output directory for this input if it doesn't exist
        # No timestamp needed here unless specifically desired for multiple runs on same input
        os.makedirs(run_output_dir, exist_ok=True) # Create dir immediately

        # --- Load Input Image --- (Moved inside loop)
        print(f"Loading input image: {input_image_path_or_url}")
        try:
            input_image = load_image(input_image_path_or_url)

            # <<< Add background removal >>>
            if rembg_session: # Only attempt if session initialized
                print("Removing background...")
                input_image = remove_background(input_image, rembg_session=rembg_session)

                # <<< Convert RGBA back to RGB if needed >>>
                if input_image.mode == 'RGBA':
                    print("Converting RGBA background-removed image to RGB...")
                    # Create a white background
                    background = Image.new("RGB", input_image.size, (255, 255, 255))
                    # Paste the RGBA image onto the white background using the alpha channel as mask
                    background.paste(input_image, mask=input_image.split()[3])
                    input_image = background
                # <<< End RGBA to RGB conversion >>>

            else:
                print("Skipping background removal due to session initialization error.")
            # <<< End background removal >>>

            target_size = (256, 256)
            print(f"Resizing input image to {target_size}...")
            input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading, removing background, or resizing input image: {e}. Skipping this input.") # Updated error message
            continue # Skip to next input image

        # --- Generation and Verification Loop (Per Input) ---
        best_group_data = {
            "avg_score": -1,
            "images": None,
            "gemini_scores": None,
            "laion_scores": None,
            "seed": -1
        }

        print(f"Generating and evaluating {num_candidates} candidate groups for {input_name}...")
        for i in range(num_candidates):
            print(f"\n--- Candidate Group {i+1}/{num_candidates} for {input_name} ---")
            # Use a unique seed for each candidate group generation
            current_seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(current_seed)

            # Define viewpoints using camera poses from camera_utils
            num_views = 6 # Keep 12 views
            radius = 3.0 # Standard radius for Zero123-Plus
            elevation = 20.0 # Standard elevation for Zero123-Plus
            print(f"Generating camera poses for {num_views} views (Radius: {radius}, Elevation: {elevation})...")
            # Get (num_views, 4, 4) camera-to-world matrices
            camera_poses = get_circular_camera_poses(M=num_views, radius=radius, elevation=elevation)
            camera_poses = camera_poses.to(device) # Move poses to the correct device

            generated_images = []

            print(f"Generating {len(camera_poses)} views with seed: {current_seed}...")
            generation_successful = True
            # Loop through generated camera poses
            for view_idx, pose in enumerate(camera_poses):
                print(f"  Generating view {view_idx+1}/{len(camera_poses)}...")
                try:
                    # Generate SINGLE view image using the current pose matrix
                    result = pipeline(
                        input_image, # Input is the background-removed, resized image
                        pose=pose,   # Pass the 4x4 pose matrix
                        num_inference_steps=28,
                        generator=generator,
                        height=320, # Zero123-Plus often uses 320x320
                        width=320,
                        # Remove elevation, azimuth, distance
                    )
                    if hasattr(result, 'images') and result.images:
                        generated_images.append(result.images[0])
                    else:
                        print(f"  Warning: Pipeline result for view {view_idx+1} did not contain expected image output.")
                        generation_successful = False
                        break
                except Exception as e:
                    print(f"  Error during image generation for view {view_idx+1}: {e}")
                    generation_successful = False
                    break

            if not generation_successful or not generated_images:
                print(f"Skipping verification for group {i+1} due to generation errors.")
                continue

            print(f"Generated {len(generated_images)} views successfully for group {i+1}.")

            # 2. Apply Gemini Verifier
            print("Applying Gemini Verifier...")
            gemini_prompts = [prompt_text] * len(generated_images)
            try:
                gemini_inputs = gemini_verifier.prepare_inputs(images=generated_images, prompts=gemini_prompts)
                gemini_results = gemini_verifier.score(inputs=gemini_inputs)
                group_overall_scores = [res.get("overall_score", {}).get("score", 0) for res in gemini_results]
                avg_group_score = np.mean(group_overall_scores) if group_overall_scores else 0
                print(f"Gemini Average Overall Score for Group: {avg_group_score:.4f}")
            except Exception as e:
                print(f"Error during Gemini verification for group {i+1}: {e}")
                avg_group_score = -1
                gemini_results = []

            # 3. Apply LAION Aesthetic Verifier
            print("Applying LAION Aesthetic Verifier...")
            try:
                laion_inputs = laion_verifier.prepare_inputs(images=generated_images)
                laion_results = laion_verifier.score(inputs=laion_inputs)
                avg_laion_score = np.mean([res["laion_aesthetic_score"] for res in laion_results])
                print(f"LAION Average Aesthetic Score for Group: {avg_laion_score:.4f}")
            except Exception as e:
                print(f"Error during LAION verification for group {i+1}: {e}")
                laion_results = []

            # 4. Check if this group is the best so far
            if avg_group_score > best_group_data["avg_score"]:
                print(f"New best group found for {input_name} with average Gemini score: {avg_group_score:.4f}")
                best_group_data["avg_score"] = avg_group_score
                best_group_data["images"] = generated_images
                best_group_data["gemini_scores"] = gemini_results
                best_group_data["laion_scores"] = laion_results
                best_group_data["seed"] = current_seed

        # --- Process and Save Best Group (Per Input) ---
        if best_group_data["images"] is None:
            print(f"\nNo successful candidate groups were generated or verified for input {input_name}.")
            continue # Skip to the next input image

        print(f"\n--- Best Group for {input_name} (Seed: {best_group_data['seed']}) --- ")
        print(f"Average Gemini Overall Score: {best_group_data['avg_score']:.4f}")

        # Save input image
        input_img_save_path = os.path.join(run_output_dir, "input_image.png")
        try:
            input_image.save(input_img_save_path)
            print(f"Saved input image to: {input_img_save_path}")
        except Exception as e:
            print(f"Error saving input image {input_img_save_path}: {e}")

        # Save generated images from the best group
        saved_image_paths = []
        best_gemini_scores = best_group_data["gemini_scores"]
        for i, img in enumerate(best_group_data["images"]):
            prompt_slug = "".join(c if c.isalnum() else '_' for c in prompt_text[:50])
            img_score = best_gemini_scores[i].get('overall_score', {}).get('score', 'N/A') if best_gemini_scores else 'N/A'
            img_score_str = f"{img_score:.1f}" if isinstance(img_score, (int, float)) else str(img_score)
            img_filename = f"prompt@{prompt_slug}_s@{best_group_data['seed']}_i@{i}_score@{img_score_str}.png"
            img_path = os.path.join(run_output_dir, img_filename)
            try:
                img.save(img_path)
                relative_img_path = os.path.relpath(img_path, start=os.getcwd())
                saved_image_paths.append(relative_img_path)
            except Exception as e:
                print(f"Error saving image {img_filename}: {e}")

        print(f"Saved {len(saved_image_paths)} generated images to: {run_output_dir}")

        # Calculate average LAION score for the best group (if available)
        best_laion_scores = best_group_data["laion_scores"]
        avg_laion_score_best_group = np.mean([res["laion_aesthetic_score"] for res in best_laion_scores]) if best_laion_scores else None

        # --- Prepare Final Output JSON (Per Input) ---
        # Extract representative Gemini explanation
        gemini_explanation = "Explanation not available"
        if best_gemini_scores and isinstance(best_gemini_scores, list) and len(best_gemini_scores) > 0:
             first_image_scores = best_gemini_scores[0]
             if isinstance(first_image_scores, dict):
                 gemini_explanation = first_image_scores.get('overall_score', {}).get('explanation', gemini_explanation)

        output_data = {
            "input_identifier": input_name,
            "input_source": input_image_path_or_url,
            "prompt": prompt_text,
            "best_score": {
                "metric": "Average Gemini 'overall_score' across all views in the best group.",
                "score": best_group_data["avg_score"],
                "representative_explanation": gemini_explanation # Added explanation
            },
            "choice_of_metric": choice_of_metric_val,
            "best_group_dir": os.path.relpath(run_output_dir, start=os.getcwd()),
            "best_group_seed": best_group_data["seed"],
            "best_group_avg_laion_score": avg_laion_score_best_group,
            # Optional: Add paths to saved generated images if needed
            # "saved_generated_image_paths": saved_image_paths
        }

        # --- Save Results JSON (Per Input) ---
        results_filename = os.path.join(run_output_dir, "results.json")
        try:
            with open(results_filename, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Results JSON saved to: {results_filename}")
        except Exception as e:
            print(f"Error saving results JSON: {e}")

    print(f"\n{'='*20} Script Finished Processing All Inputs {'='*20}")
