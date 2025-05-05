from google import genai
from google.genai import types
import json
import os
import sys
from typing import List, Dict, Any
from PIL import Image
import base64
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from verifiers.base_verifier import BaseVerifier


class GeminiVerifier(BaseVerifier):
    def __init__(self, seed=1994, model_name="gemini-2.0-flash", **kwargs):
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_config = types.GenerateContentConfig(
            system_instruction=self.verifier_prompt,
            response_mime_type="application/json",
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
            
            # Add the prompt as text with user role
            content_parts.append(types.Content(
                role="user",
                parts=[types.Part(text=self.verifier_prompt)]
            ))
            
            # Add the images with user role
            for img_part in inputs["images"]:
                content_parts.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        inline_data=types.Blob(
                            mime_type=img_part["mime_type"],
                            data=img_part["data"]
                        )
                    )]
                ))
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=self.generation_config
            )
            
            # Parse the response
            try:
                # Get the response text
                response_text = response.text
                print(f"Raw response from Gemini: {response_text}")  # Debug print
                
                # Try to parse as JSON directly first
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from the text
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found in response")
                
                # Ensure the result has the expected structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Add default scores if missing
                default_scores = {
                    "aesthetic_quality": {"score": 0, "explanation": "No score provided"},
                    "visual_consistency": {"score": 0, "explanation": "No score provided"},
                    "reconstruction_potential": {"score": 0, "explanation": "No score provided"},
                    "overall_score": 0,
                    "overall_assessment": "No assessment provided"
                }
                
                # Update with actual scores if present
                for key in default_scores:
                    if key in result:
                        if isinstance(result[key], dict):
                            default_scores[key].update(result[key])
                        else:
                            default_scores[key] = result[key]
                
                # Calculate overall score if not provided
                if default_scores["overall_score"] == 0:
                    scores = [
                        default_scores["aesthetic_quality"]["score"],
                        default_scores["visual_consistency"]["score"],
                        default_scores["reconstruction_potential"]["score"]
                    ]
                    default_scores["overall_score"] = sum(scores) / len(scores)
                
                return {
                    "success": True,
                    "result": default_scores,
                    "raw_response": response_text
                }
                
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                print(f"Raw response: {response_text}")
                return {
                    "success": False,
                    "error": f"Failed to parse response: {str(e)}",
                    "raw_response": response_text
                }
                
        except Exception as e:
            print(f"Error during Gemini evaluation: {str(e)}")
            return {
                "success": False,
                "error": f"Error during Gemini evaluation: {str(e)}",
                "raw_response": None
            }