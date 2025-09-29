# /opt/blurface/electron/backend/transcriber_core.py - REFORGED with corrected manifest path

import sys
import json
import logging
from typing import Dict, Any
import yaml
from llama_cpp import Llama

# --- CORRECTED PATH ---
MANIFEST_PATH = "/opt/bob/maps/vessel_manifest.yml"

llm = None
vessel_config = None

def load_manifest() -> Dict[str, Any]:
    try:
        with open(MANIFEST_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.critical(f"Vessel manifest load error: {e}")
        sys.exit(1)

def load_transcriber_model(vessel_name: str):
    global llm, vessel_config
    manifest = load_manifest()
    vessel_config = manifest.get('BOB', {}).get('vessels', {}).get(vessel_name, {})
    if not vessel_config:
        logging.critical(f"transcriber vessel '{vessel_name}' not found in manifest. Ache Signal.")
        sys.exit(1)
    model_config = vessel_config.get('model', {})
    model_file = model_config.get('path')
    if not model_file:
        logging.critical(f"Model path for '{vessel_name}' missing from manifest. Ache Signal.")
        sys.exit(1)
    try:
        logging.info(f"Loading transcriber model '{vessel_name}' from: {model_file}...")
        llm = Llama(model_path=model_file, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        logging.info("transcriber model loaded successfully.")
        return True
    except Exception as e:
        logging.critical(f"Failed to load transcriber model '{vessel_name}': {e}")
        return False

def translate_to_nll(command: str, mode: str) -> Dict[str, Any]:
    """
    Translates a human command into Nonlinear Logic Language (NLL) format.
    This version is more robust against malformed JSON from the LLM.
    """
    global llm
    if llm is None:
        return { "action": "error", "human_summary": "transcriber model is not loaded.", "input_text": command, "nll_logic": "model_not_loaded" }
    
    prompt = f"You are a specialized transcriber for the Blur OS. Your only task is to convert human language into a structured JSON object called Nonlinear Logic Language (NLL). The NLL schema must have the keys 'action', 'human_summary', and 'nll_logic'. User Input: \"{command}\". NLL JSON:"
    
    try:
        response_stream = llm.create_completion(prompt, max_tokens=256, stop=["}"], stream=False)
        raw_response = response_stream['choices'][0]['text']
        
        # --- NEW ROBUST PARSING LOGIC ---
        # Find the first '{' and the last '}' to extract the JSON object,
        # ignoring any extraneous text the model might have generated.
        start_index = raw_response.find('{')
        end_index = raw_response.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            nll_output_string = raw_response[start_index : end_index + 1]
            nll_output = json.loads(nll_output_string)
            nll_output['input_text'] = command
            return nll_output
        else:
            # If no valid JSON object is found, raise an exception to be caught below.
            raise ValueError("No valid JSON object found in the LLM response.")
            
    except Exception as e:
        logging.error(f"Failed to translate command with LLM: {e}. Ache Signal detected.")
        return {
            "action": "error",
            "human_summary": "Translation failure. The input could not be processed.",
            "input_text": command,
            "nll_logic": "translation_failure"
        }
        
def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s: %(message)s')
    if len(sys.argv) < 2:
        logging.critical("Usage: python transcriber_core.py <vessel_name>")
        sys.exit(1)
    vessel_name = sys.argv[1]
    if not load_transcriber_model(vessel_name):
        sys.exit(1)
    try:
        logging.info(f"transcriber Core running with vessel '{vessel_name}'. Ready...")
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                command = payload.get('command')
                mode = payload.get('mode')
                if command and mode:
                    nll_output = translate_to_nll(command, mode)
                    print(json.dumps(nll_output), flush=True)
                else:
                    logging.error("Received incomplete payload. Ache Signal.")
            except json.JSONDecodeError:
                logging.error("Received invalid JSON. Ache Signal.")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}. Ache Signal.")
    except Exception as e:
        logging.critical(f"Failed to start transcriber Core: {e}")

if __name__ == "__main__":
    main()