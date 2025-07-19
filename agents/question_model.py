# Starting with Qwen3-4B in action.
import time
import torch
from typing import Optional, Union, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class QAgent(object):
    def __init__(self, **kwargs):
        # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
        self.model_type = kwargs.get('model_type', '4B').strip()
        # Use valid HuggingFace model path instead of local path
        model_name = "Qwen/Qwen2.5-3B-Instruct"  # Valid HF model
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(self, message: str|List[str], system_prompt: Optional[str] = None, **kwargs)->str:
        """Generate response with thinking mode enabled only for question creation"""
        
        # Enable thinking mode for question creation
        enable_thinking_mode = True
        
        # Handle both single message and batch
        if isinstance(message, str):
            messages = [message]
        else:
            messages = message
        
        # Build conversation for each message
        conversations = []
        for msg in messages:
            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            conversation.append({"role": "user", "content": msg})
            conversations.append(conversation)
        
        # Apply chat template with thinking mode
        model_inputs_list = []
        for conversation in conversations:
            model_input = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt",
                enable_thinking=enable_thinking_mode  # Enable thinking mode
            )
            model_inputs_list.append(model_input)
        
        # Batch the inputs
        max_length = max(input_ids.shape[1] for input_ids in model_inputs_list)
        batched_input_ids = []
        batched_attention_mask = []
        
        for input_ids in model_inputs_list:
            # Pad to max length
            pad_length = max_length - input_ids.shape[1]
            if pad_length > 0:
                padding = torch.full((1, pad_length), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                input_ids = torch.cat([padding, input_ids], dim=1)
            
            attention_mask = (input_ids != (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)).long()
            batched_input_ids.append(input_ids)
            batched_attention_mask.append(attention_mask)
        
        # Stack into batch tensors
        model_inputs = {
            'input_ids': torch.cat(batched_input_ids, dim=0),
            'attention_mask': torch.cat(batched_attention_mask, dim=0)
        }
        
        # Set generation parameters
        tgps_show_var = kwargs.get("tgps_show", False)
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.6),  # Recommended for thinking mode
            "top_p": kwargs.get("top_p", 0.95),  # Recommended for thinking mode
            "top_k": kwargs.get("top_k", 20),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                **generation_kwargs
            )
        
        generation_time = time.time() - start_time
        
        # Process outputs and extract thinking content
        batch_outs = []
        token_len = 0
        
        thinking_token_id = 151668  # </think> token ID
        
        for i, output_ids in enumerate(generated_ids):
            # Remove input tokens to get only generated content
            input_length = len(model_inputs['input_ids'][i])
            output_ids = output_ids[input_length:]
            token_len += len(output_ids)
            
            # Extract final answer after thinking
            final_answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            if enable_thinking_mode and thinking_token_id in output_ids:
                # Find the thinking end token
                thinking_positions = (output_ids == thinking_token_id).nonzero(as_tuple=True)[0]
                if len(thinking_positions) > 0:
                    thinking_end_pos = thinking_positions[-1].item()  # Last occurrence
                    
                    # Extract final answer (after </think>)
                    final_tokens = output_ids[thinking_end_pos + 1:]
                    final_answer = self.tokenizer.decode(final_tokens, skip_special_tokens=True).strip()
            
            # Try to extract JSON if it's malformed
            if final_answer and not final_answer.startswith('{'):
                # Look for JSON in the response
                json_start = final_answer.find('{')
                if json_start != -1:
                    json_end = final_answer.rfind('}')
                    if json_end != -1:
                        final_answer = final_answer[json_start:json_end + 1]
            
            batch_outs.append(final_answer)
        
        # Return based on tgps_show setting
        if tgps_show_var:
            if len(batch_outs) == 1:
                return batch_outs[0], token_len, generation_time
            else:
                return batch_outs, token_len, generation_time
        else:
            if len(batch_outs) == 1:
                return batch_outs[0]
            else:
                return batch_outs
