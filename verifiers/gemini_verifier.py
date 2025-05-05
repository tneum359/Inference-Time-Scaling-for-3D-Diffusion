from google import genai
from google.genai import types
import typing_extensions as typing
import json
import os
import sys
from typing import Union, List, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)


from verifiers.base_verifier import BaseVerifier
from utils import convert_to_bytes


class Score(typing.TypedDict):
    explanation: str
    score: float


class Grading(typing.TypedDict):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class GeminiVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name="gemini-2.0-flash", **kwargs):
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_config = types.GenerateContentConfig(
            system_instruction=self.verifier_prompt,
            response_mime_type="application/json",
            response_schema=list[Grading],
            max_output_tokens=kwargs.pop("max_new_tokens", None),
            seed=seed,
        )
        self.model_name = model_name

    def prepare_inputs(self, images: List[Image.Image], prompts: List[str]) -> Dict[str, Any]:
        """
        Prepare inputs for the Gemini model.
        
        Args:
            images: List of PIL Images (should be 6 multiview images)
            prompts: List of prompts (should be the same prompt repeated 6 times)
            
        Returns:
            Dictionary containing the prepared inputs
        """
        if len(images) != 6:
            raise ValueError("Expected exactly 6 multiview images")
            
        # Convert images to base64
        image_parts = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_parts.append({
                "mime_type": "image/png",
                "data": img_str
            })
            
        return {
            "images": image_parts,
            "prompt": self.verifier_prompt
        }

    def score(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score the multiview images using Gemini.
        
        Args:
            inputs: Dictionary containing images and prompt
            
        Returns:
            Dictionary containing the evaluation results
        """
        try:
            # Create the content parts for the model
            content_parts = []
            
            # Add the prompt
            content_parts.append(self.verifier_prompt)
            
            # Add the images
            for img_part in inputs["images"]:
                content_parts.append({
                    "mime_type": img_part["mime_type"],
                    "data": img_part["data"]
                })
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name, contents=content_parts, config=self.generation_config
            )
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text
                # Find the JSON object in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response")
                
                return {
                    "success": True,
                    "result": result,
                    "raw_response": response_text
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during Gemini evaluation: {str(e)}"
            }


# Define inputs
if __name__ == "__main__":
    verifier = GeminiVerifier()
    image_urls = [
        (
            "realistic photo a shiny black SUV car with a mountain in the background.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        ),
        (
            "photo a green and funny creature standing in front a lightweight forest.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/green_creature.jpg",
        ),
    ]

    prompts = []
    images = []
    for text, path_or_url in image_urls:
        prompts.append(text)
        images.append(path_or_url)

    # # Single image
    # response = client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     contents=[
    #         "realistic photo a shiny black SUV car with a mountain in the background.",
    #         load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg")
    #     ],
    #     config=generation_config
    # )
    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)

    with open("results.json", "w") as f:
        json.dump(response, f)

    print(json.dumps(response, indent=4))