from google import genai
from google.genai import types
import json
import os
import sys
from typing import List, Dict, Any
from PIL import Image
import base64
from io import BytesIO
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from verifiers.base_verifier import BaseVerifier


class GeminiVerifier(BaseVerifier):
    def __init__(self, seed=1994, model_name="gemini-1.5-flash", **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=model_name)
        self.model_name = model_name
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            # stop_sequences=['}'], # Could cause issues if explanation contains '}'
            # max_output_tokens=2048, # Default is often sufficient
            temperature=0.7, # Adjust for creativity vs. consistency
            # top_p=1.0,
            # top_k=40,
            response_mime_type="application/json", # Request JSON output directly
        )
        self.verifier_prompt = kwargs.get('gemini_prompt', "No prompt provided.")
        print(f"GeminiVerifier initialized with model: {model_name}")
        # print(f"Using prompt: {self.verifier_prompt[:100]}...") # Print start of prompt

    # This method is no longer directly called by run.py for formatting,
    # but its logic is now integrated into the score method's caller in run.py
    # and the score method here will assemble the final API payload.
    def prepare_inputs(self, original_image_pil: Image.Image, candidate_images_pil: List[Image.Image]) -> Dict[str, Any]:
        """
        Prepares the original image and candidate images into a dictionary with base64 encoded data.
        This structure will be used by the score method.
        """
        prepared_data = {
            "original_image_b64": None,
            "candidate_views_b64": []
        }
        try:
            # Process original image
            buffered = BytesIO()
            original_image_pil.save(buffered, format="PNG")
            prepared_data["original_image_b64"] = base64.b64encode(buffered.getvalue()).decode()

            # Process candidate images
            for img_pil in candidate_images_pil:
                buffered = BytesIO()
                img_pil.save(buffered, format="PNG")
                prepared_data["candidate_views_b64"].append(base64.b64encode(buffered.getvalue()).decode())
        except Exception as e:
            print(f"Error during image preparation for Gemini: {e}")
            # Return partially filled data or raise error, depending on desired handling
        return prepared_data

    def score(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score the multiview images using Gemini, considering the original input image.
        
        Args:
            inputs: Dictionary containing:
                'original_image_b64': base64 string of the original input image.
                'candidate_views_b64': list of base64 strings for the 6 candidate multiviews.
                'prompt': The instructional prompt string (though self.verifier_prompt is used).
            
        Returns:
            Dictionary containing the evaluation results
        """
        try:
            if not inputs.get("original_image_b64") or not inputs.get("candidate_views_b64"):
                print("Error: Missing original_image_b64 or candidate_views_b64 in inputs to score method.")
                return {"success": False, "error": "Missing image data in inputs.", "raw_response": None}

            # Assemble content parts for the Gemini API
            api_content_parts = [
                types.Part(text=self.verifier_prompt), # The main prompt
                types.Part(text="\n\n--- Initial Input Image ---"),
                types.Part(inline_data=types.Blob(
                    mime_type="image/png",
                    data=inputs["original_image_b64"]
                )),
                types.Part(text="\n\n--- Candidate Multiview Set ---")
            ]
            
            for view_b64_data in inputs["candidate_views_b64"]:
                api_content_parts.append(types.Part(inline_data=types.Blob(
                    mime_type="image/png", # Assuming PNG as per prepare_inputs logic
                    data=view_b64_data
                )))
            
            print(f"DEBUG: Gemini API Content Parts being sent: {api_content_parts}")

            # Create the single Content object for the API call
            request_contents = [types.Content(role="user", parts=api_content_parts)]
            
            # Generate response
            print(f"Sending request to Gemini ({self.model_name})... Content parts: {len(api_content_parts)}")
            response = self.client.generate_content(
                model=self.model_name,
                contents=request_contents,
                generation_config=self.generation_config
            )
            
            # Parse the response
            try:
                response_text = response.text
                print(f"Raw response from Gemini: {response_text[:500]}...") # Print more of the response
                
                # The response should be JSON directly due to response_mime_type="application/json"
                result = json.loads(response_text)
                                
                # Ensure the result has the expected structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Add default scores, now including consistency_with_input_image
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
                        if isinstance(result[key], dict) and "score" in result[key] and "explanation" in result[key]:
                            default_scores[key]["score"] = result[key]["score"]
                            default_scores[key]["explanation"] = result[key]["explanation"]
                        elif isinstance(result[key], (int, float)) and key == "overall_score": # Handle overall_score directly if it's just a number
                             default_scores[key] = result[key]
                        elif key != "overall_score": # For sub-dictionaries like aesthetic_quality
                             print(f"Warning: Unexpected format for '{key}' in Gemini response. Received: {result[key]}")
                
                # Calculate overall_score if not robustly provided or to enforce logic
                # For now, let's trust Gemini's overall_score if present and valid, otherwise average the main 4.
                if not isinstance(default_scores["overall_score"], (int, float)) or default_scores["overall_score"] == 0:
                    print("Calculating overall_score as average of component scores.")
                    scores_for_avg = [
                        default_scores["consistency_with_input_image"]["score"],
                        default_scores["aesthetic_quality"]["score"],
                        default_scores["visual_consistency_across_views"]["score"],
                        default_scores["reconstruction_potential"]["score"]
                    ]
                    # Filter out non-numeric if any somehow slipped through (shouldn't with init)
                    numeric_scores_for_avg = [s for s in scores_for_avg if isinstance(s, (int, float))]
                    if numeric_scores_for_avg:
                         default_scores["overall_score"] = sum(numeric_scores_for_avg) / len(numeric_scores_for_avg)
                    else:
                         default_scores["overall_score"] = 0 # Fallback if no numeric component scores
                
                return {
                    "success": True,
                    "result": default_scores,
                    "raw_response": response_text
                }
                
            except json.JSONDecodeError as e_json:
                print(f"Error parsing JSON response from Gemini: {e_json}")
                # Try to extract JSON from markdown if present (common issue)
                if "```json" in response_text:
                    try:
                        json_str = response_text.split("```json\n")[1].split("\n```")[0]
                        print(f"Extracted JSON from markdown: {json_str[:200]}...")
                        result_from_markdown = json.loads(json_str)
                        # Re-run the default score population with this extracted dict
                        # This part is a bit repetitive, could be refactored into a helper
                        populated_result = default_scores.copy() # Start fresh
                        for key in populated_result:
                            if key in result_from_markdown:
                                if isinstance(result_from_markdown[key], dict) and "score" in result_from_markdown[key]:
                                    populated_result[key].update(result_from_markdown[key])
                                elif isinstance(result_from_markdown[key], (int, float)) and key == "overall_score":
                                    populated_result[key] = result_from_markdown[key]
                        # Recalculate overall if needed
                        if not isinstance(populated_result["overall_score"], (int, float)) or populated_result["overall_score"] == 0:
                            scores_for_avg = [populated_result[k]["score"] for k in ["consistency_with_input_image", "aesthetic_quality", "visual_consistency_across_views", "reconstruction_potential"] if isinstance(populated_result[k], dict) and isinstance(populated_result[k].get("score"), (int, float))]
                            if scores_for_avg: populated_result["overall_score"] = sum(scores_for_avg) / len(scores_for_avg)
                            else: populated_result["overall_score"] = 0
                        return {"success": True, "result": populated_result, "raw_response": response_text}
                    except Exception as e_extract:
                        print(f"Error extracting/parsing JSON from markdown: {e_extract}")
                # Fall through to generic error if markdown extraction fails or not present
                return {"success": False, "error": f"Failed to parse response (JSONDecodeError): {e_json}", "raw_response": response_text}
            except Exception as e_parse:
                print(f"Error processing/parsing Gemini response: {str(e_parse)}")
                # response_text might not be defined if error was before .text access
                raw_resp_text = response.text if hasattr(response, 'text') else "Response object had no text attribute (request may have failed earlier)"
                print(f"Raw response (if available): {raw_resp_text[:500]}...")
                return {"success": False, "error": f"Failed to process/parse response: {str(e_parse)}", "raw_response": raw_resp_text}
                
        except Exception as e_eval:
            print(f"Error during Gemini API call: {str(e_eval)}")
            traceback.print_exc()
            return {"success": False, "error": f"Error during Gemini API call: {str(e_eval)}", "raw_response": None}