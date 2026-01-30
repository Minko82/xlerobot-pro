#!/usr/bin/env python3
"""
Server Script (GPU Side) - FIXED BOOLEAN MASK & DUAL CAMERA
"""

import asyncio
import logging
import websockets
import msgpack
import torch
import numpy as np
import cv2

# LeRobot & Transformers Imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor

# Setup Logging
logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s - SERVER - %(levelname)s - %(message)s')
logger = logging.getLogger("PolicyServer")

# Global variables
policy_model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_policy():
    """Load the SmolVLA model AND processor into memory."""
    global policy_model, processor
    logger.info(f"🧠 Loading SmolVLA model on {device}...")
    
    pretrained_path = "lerobot/smolvla_base" 
    
    # 1. Load Policy
    policy_model = SmolVLAPolicy.from_pretrained(pretrained_path)
    policy_model.eval()
    policy_model.to(device)
    
    # 2. Load Processor
    logger.info("📝 Loading Processor...")
    try:
        processor = AutoProcessor.from_pretrained(pretrained_path)
    except Exception:
        # Fallback if specific processor config is missing
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    logger.info("✅ Model and Processor loaded.")

def decode_image(img_bytes):
    """Decodes raw bytes, RESIZES to 256x256, and converts to Tensor."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize to 256x256 (Required by SmolVLA)
    frame = cv2.resize(frame, (256, 256))
    
    # Convert to Tensor (C, H, W) and normalize
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return img_tensor.to(device)

async def handler(websocket):
    logger.info("🔗 Client connected.")

    # Metadata Handshake
    metadata = {
        "action_space": {
            "shape": [6],
            "dtype": "float32"
        },
        "observation_space": {
            "image": {"shape": [3, 256, 256], "dtype": "uint8"},
            "state": {"shape": [6], "dtype": "float32"}
        },
        "protocol_version": 1
    }
    await websocket.send(msgpack.packb(metadata))
    logger.info("📤 Sent metadata handshake.")

    try:
        async for message in websocket:
            payload = msgpack.unpackb(message, raw=False)
            
            # 1. Process Images (Dual Camera)
            # Check for new dual-camera keys, otherwise fall back to single
            if 'image_top' in payload and 'image_wrist' in payload:
                img_tensor_top = decode_image(payload['image_top'])
                img_tensor_wrist = decode_image(payload['image_wrist'])
            else:
                # Fallback for old clients
                img_raw = payload.get('image', {}).get('bytes', payload.get('image'))
                img_tensor_top = decode_image(img_raw)
                img_tensor_wrist = img_tensor_top # Duplicate if missing (or handle appropriately)

            img_tensor_top = img_tensor_top.unsqueeze(0)
            img_tensor_wrist = img_tensor_wrist.unsqueeze(0)

            # 2. Process State
            state_tensor = torch.tensor(payload['state']).float().unsqueeze(0).to(device)
            
            # 3. Process Text
            text_prompt = payload.get('prompt', "do something")
            
            # Tokenize: generate inputs_ids AND attention_mask
            text_inputs = processor(text=text_prompt, return_tensors="pt")
            
            input_ids = text_inputs.input_ids.to(device)
            
            # --- FIX: Convert mask to BOOL ---
            # The model requires a boolean mask, but tokenizer gives 0/1 integers
            attention_mask = text_inputs.attention_mask.to(device).bool()
            # ---------------------------------

            # 4. Construct Batch
            batch = {
                "observation.images.camera1": img_tensor_top,   # Top View
                "observation.images.camera2": img_tensor_wrist, # Wrist View
                "observation.state": state_tensor,
                "observation.language.tokens": input_ids,
                "observation.language.attention_mask": attention_mask
            }

            with torch.inference_mode():
                action_tensor = policy_model.select_action(batch)
            
            action_list = action_tensor.squeeze(0).cpu().numpy().tolist()
            
            response = {"actions": action_list}
            await websocket.send(msgpack.packb(response))

    except websockets.exceptions.ConnectionClosed:
        logger.info("❌ Client disconnected.")
    except Exception as e:
        logger.error(f"Error during inference: {e}")

async def main():
    load_policy()
    async with websockets.serve(handler, "0.0.0.0", 8000):
        logger.info("🚀 Server listening on 0.0.0.0:8000")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass