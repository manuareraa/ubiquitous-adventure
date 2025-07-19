# Qwen3-4B in action.
import time
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)
class AAgent(object):
    def __init__(self, **kwargs):
        # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
        self.model_type = kwargs.get('model_type', '4B').strip()
        # model_name = "Qwen/Qwen3-4B"
        model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(self, message: str|List[str], system_prompt: Optional[str] = None, **kwargs)->str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        
        # Prepare all messages for batch processing using chat template with thinking mode
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ]
            all_messages.append(messages)
        
        # Convert messages to text format using chat template with thinking mode enabled
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # Enable thinking mode by default
            )
            texts.append(text)
        
        # Tokenize all texts
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        
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
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                **generation_kwargs
            )
        
        generation_time = time.time() - start_time
        
        # Process outputs and extract thinking content
        batch_outs = []
        token_len = 0
        
        for i, output_ids in enumerate(generated_ids):
            # Remove input tokens to get only generated content
            input_length = len(model_inputs.input_ids[i])
            output_ids = output_ids[input_length:].tolist()
            token_len += len(output_ids)
            
            # Extract thinking content and final response
            try:
                # Find the end of thinking block (token 151668 is </think>)
                index = len(output_ids) - output_ids[::-1].index(151668) if 151668 in output_ids else 0
            except ValueError:
                index = 0
            
            # Decode thinking content (if any)
            thinking_content = ""
            if index > 0:
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            
            # Decode final response content
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # For debugging/verbose mode, you can print thinking content
            if kwargs.get("show_thinking", False) and thinking_content:
                print(f"🧠 [THINKING]: {thinking_content[:200]}...")
            
            batch_outs.append(content)
        
        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None
        
if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response("Solve: 2x + 5 = 15", system_prompt="You are a math tutor.", tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print(f"Single response: {response}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")
          
    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(messages, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True, tgps_show=True)
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", 
        temperature=0.8, 
        max_new_tokens=512
    )
    print(f"Custom response: {response}")
