# Starting with Qwen3-4B in action.
import time
import torch
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class QAgent(object):
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
        # Enhanced system prompt based on original file approach
        sys_prompt = """You are an **expert-level examiner** with deep expertise in designing **highly challenging and conceptually rigorous multiple-choice questions (MCQs)** for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier competitive exams.

Think step by step to generate the question and solve the same, but only output the final JSON answer. Do not show your thinking process.

**CRITICAL: Please DO NOT reveal the solution steps or any intermediate reasoning in your final output. Output ONLY the JSON object.**

**STYLE MATCHING**: Follow the exact style, complexity, vocabulary, and structure of the provided examples. Match their tone, difficulty level, and presentation format."""
        if system_prompt is None:
            system_prompt = sys_prompt
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
        
        # Convert messages to text format using chat template with selective thinking mode
        texts = []
        # Control thinking mode - enable only for question creation, not for choices/post-processing
        enable_thinking_mode = kwargs.get("enable_thinking", True)  # Allow override
        thinking_stage = kwargs.get("thinking_stage", "question_creation")  # question_creation, distractor_generation, validation
        
        # Restrict thinking to question creation only (as requested by user)
        if thinking_stage != "question_creation":
            enable_thinking_mode = False
            print(f"🧠 [THINKING] Disabled for stage: {thinking_stage}")
        else:
            print(f"🧠 [THINKING] Enabled for stage: {thinking_stage}")
            
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_mode
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
        
        thinking_token_id = 151668  # </think> token ID
        
        for i, output_ids in enumerate(generated_ids):
            # Remove input tokens to get only generated content
            input_length = len(model_inputs.input_ids[i])
            output_ids = output_ids[input_length:]
            token_len += len(output_ids)
            
            # Extract thinking content and final answer
            thinking_content = ""
            final_answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            if enable_thinking_mode and thinking_token_id in output_ids:
                # Find the thinking end token
                thinking_positions = (output_ids == thinking_token_id).nonzero(as_tuple=True)[0]
                if len(thinking_positions) > 0:
                    thinking_end_pos = thinking_positions[-1].item()  # Last occurrence
                    
                    # Extract thinking content (before </think>)
                    thinking_tokens = output_ids[:thinking_end_pos]
                    thinking_content = self.tokenizer.decode(thinking_tokens, skip_special_tokens=True)
                    
                    # Extract final answer (after </think>)
                    final_tokens = output_ids[thinking_end_pos + 1:]
                    final_answer = self.tokenizer.decode(final_tokens, skip_special_tokens=True).strip()
                    
                    if kwargs.get("show_thinking", False) and thinking_content:
                        print(f" [THINKING] Model reasoning:\n{thinking_content}")
                        print(f" [FINAL] Model answer:\n{final_answer}")
            
            # JSON Extraction and Self-Correction (based on original file)
            if final_answer:
                try:
                    # Try to parse as JSON first
                    json.loads(final_answer)
                    print(" [JSON] Valid JSON output detected")
                except json.JSONDecodeError as e:
                    print(f" [JSON] Invalid JSON format detected: {e}")
                    print(" [JSON] Attempting self-correction...")
                    
                    # Use self-correction prompt (from original file)
                    correction_prompt = (
                        'Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n'
                        'Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n'
                        
                        'String:\n'
                        '{}\n\n'

                        'Given Format:\n'
                        '{{\n'
                        '  "topic": "...",\n'
                        '  "question": "...",\n'
                        '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                        '  "answer": "Only the option letter (A, B, C, or D)",\n'
                        '  "explanation": "..."\n'
                        '}}'
                    )
                    
                    # Generate corrected JSON without thinking mode
                    corrected_output = self.generate_response(
                        correction_prompt.format(final_answer),
                        "You are an expert JSON extractor.",
                        enable_thinking=False,
                        thinking_stage="json_correction",
                        max_new_tokens=1024,
                        temperature=0.0,
                        do_sample=False
                    )
                    
                    try:
                        json.loads(corrected_output)
                        final_answer = corrected_output
                        print(" [JSON] Self-correction successful")
                    except json.JSONDecodeError:
                        print(" [JSON] Self-correction failed, returning original")
            
            batch_outs.append(final_answer)
        
        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None

if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """
    
    response, tl, tm = model.generate_response(prompt, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print("Single example response:")
    print("Response: ", response)
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(prompts, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
