from google import generativeai as genai
from google.generativeai.types import content_types
import json
import os
import sys
import traceback
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
    def __init__(self, seed=1994, model_name="gemini-1.5-pro-vision", **kwargs):
        # Extract gemini_prompt before calling super().__init__
        self.verifier_prompt = kwargs.pop('gemini_prompt', "No prompt provided.")
        
        # Now call super() with the remaining kwargs (without gemini_prompt)
        super().__init__(seed=seed, **kwargs)
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": 0.7,
            "candidate_count": 1,
            "max_output_tokens": 2048,
        }
        
        print(f"GeminiVerifier initialized with model: {model_name}")

    def score(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Score the multiview images using Gemini, considering the original input image."""
        try:
            if not inputs.get("original_image_b64") or not inputs.get("candidate_views_b64"):
                print("Error: Missing original_image_b64 or candidate_views_b64 in inputs to score method.")
                return {"success": False, "error": "Missing image data in inputs.", "raw_response": None}

            # Create the content parts for the API call
            content_parts = []
            
            # Add the main prompt
            content_parts.append({"text": self.verifier_prompt})
            
            # Add marker and original image
            content_parts.append({"text": "\n\n--- Initial Input Image ---"})
            try:
                original_img_bytes = base64.b64decode(inputs["original_image_b64"])
                content_parts.append({
                    "mime_type": "image/png",
                    "data": original_img_bytes
                })
            except Exception as e:
                print(f"Error processing original image: {e}")
                return {"success": False, "error": f"Failed to process original image: {e}", "raw_response": None}
            
            # Add marker and candidate views
            content_parts.append({"text": "\n\n--- Candidate Multiview Set ---"})
            for i, view_b64_data in enumerate(inputs["candidate_views_b64"]):
                try:
                    view_bytes = base64.b64decode(view_b64_data)
                    content_parts.append({
                        "mime_type": "image/png",
                        "data": view_bytes
                    })
                except Exception as e:
                    print(f"Error processing candidate view {i}: {e}")
                    return {"success": False, "error": f"Failed to process candidate view {i}: {e}", "raw_response": None}

            print(f"Sending request to Gemini ({self.model_name})... Content parts: {len(content_parts)}")
            
            # Generate response
            try:
                response = self.model.generate_content(
                    contents=content_parts,
                    generation_config=self.generation_config
                )
                
                # Parse the response
                try:
                    response_text = response.text
                    print(f"Raw response from Gemini: {response_text[:500]}...")
                    
                    # Try to parse as JSON
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If direct JSON parsing fails, try to extract JSON from markdown
                        if "```json" in response_text:
                            json_str = response_text.split("```json\n")[1].split("\n```")[0]
                            result = json.loads(json_str)
                        else:
                            raise ValueError("Response is not valid JSON and couldn't extract from markdown")
                    
                    # Ensure the result has the expected structure
                    if not isinstance(result, dict):
                        raise ValueError("Response is not a dictionary")
                    
                    # Add default scores
                    default_scores = {
                        "consistency_with_input_image": {"score": 0, "explanation": "No score provided"},
                        "aesthetic_quality": {"score": 0, "explanation": "No score provided"},
                        "visual_consistency_across_views": {"score": 0, "explanation": "No score provided"},
                        "reconstruction_potential": {"score": 0, "explanation": "No score provided"},
                        "overall_score": 0,
                        "overall_assessment": "No assessment provided"
                    }
                    
                    # Update with actual scores if present
                    for key in default_scores:
                        if key in result:
                            if isinstance(result[key], dict) and "score" in result[key]:
                                default_scores[key].update(result[key])
                            elif isinstance(result[key], (int, float)) and key == "overall_score":
                                default_scores[key] = result[key]
                    
                    # Calculate overall_score if needed
                    if not isinstance(default_scores["overall_score"], (int, float)) or default_scores["overall_score"] == 0:
                        scores = [
                            default_scores[k]["score"] 
                            for k in ["consistency_with_input_image", "aesthetic_quality", "visual_consistency_across_views", "reconstruction_potential"]
                            if isinstance(default_scores[k], dict) and isinstance(default_scores[k].get("score"), (int, float))
                        ]
                        if scores:
                            default_scores["overall_score"] = sum(scores) / len(scores)
                    
                    return {
                        "success": True,
                        "result": default_scores,
                        "raw_response": response_text
                    }
                    
                except Exception as e:
                    print(f"Error processing Gemini response: {e}")
                    traceback.print_exc()
                    return {"success": False, "error": f"Failed to process response: {e}", "raw_response": response_text if 'response_text' in locals() else None}
                
            except Exception as e:
                print(f"Error during Gemini API call: {e}")
                traceback.print_exc()
                return {"success": False, "error": f"Error during Gemini API call: {e}", "raw_response": None}
                
        except Exception as e:
            print(f"Error in score method: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"Error in score method: {e}", "raw_response": None}