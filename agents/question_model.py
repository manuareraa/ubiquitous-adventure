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
        # model_name = "Qwen/Qwen3-4B"
        model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def build_prompt(self, topic: str, **kwargs) -> Tuple[str, str]:
        """Build prompt for question generation with ICL examples"""
        
        # Get ICL examples for this topic
        inc_samples = kwargs.get('inc_samples', {})
        topic_key = topic.split('/')[-1] if '/' in topic else topic
        
        print(f"🔍 [ICL] Looking for examples for topic: '{topic_key}'")
        
        # Safe lookup with fallback and verbose logging
        examples = []
        if inc_samples and topic_key in inc_samples:
            examples = inc_samples[topic_key]
            print(f"✅ [ICL] Found {len(examples)} examples for '{topic_key}'")
            if examples:
                first_q = examples[0].get('question', '')[:100]
                print(f"📝 [ICL] First example question: {first_q}...")
        else:
            available_keys = list(inc_samples.keys()) if inc_samples else []
            print(f"⚠️ [ICL] No examples found for '{topic_key}' - available keys: {available_keys}")
        
        # Build ICL examples string
        icl_examples = ""
        if examples:
            for i, sample in enumerate(examples[:3]):  # Limit to 3 examples
                icl_examples += f"\nEXAMPLE {i+1}:\n"
                icl_examples += json.dumps(sample, indent=2)
                icl_examples += "\n"
            print(f"📝 [ICL] ICL examples included: {len(icl_examples)} characters")
        else:
            print(f"⚠️ [PROMPT] No ICL examples included in prompt")
        
        # Enhanced system prompt based on original file approach
        sys_prompt = """You are an **expert-level examiner** with deep expertise in designing **highly challenging and conceptually rigorous multiple-choice questions (MCQs)** for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier competitive exams.

Think step by step to generate the question and solve the same, but only output the final JSON answer. Do not show your thinking process.

**CRITICAL: Please DO NOT reveal the solution steps or any intermediate reasoning in your final output. Output ONLY the JSON object.**

**STYLE MATCHING**: Follow the exact style, complexity, vocabulary, and structure of the provided examples. Match their tone, difficulty level, and presentation format."""
        
        # Remove model's preferential bias for options
        import random
        correct_option = random.choice(['A', 'B', 'C', 'D'])
        distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])
        
        # Enhanced prompt template based on original file structure
        prompt_template = f"""Generate an EXTREMELY DIFFICULT MCQ on topic: {topic}.

**CRITICAL REQUIREMENTS:**
1. **Topic Alignment**: The "question" must be strictly relevant to the topic: {topic}.
2. **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.
3. **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".
4. **Single Correct Answer**: Ensure that option {correct_option} is the only factually correct answer.
5. **Plausible Distractors**: Options {distractors} must be three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic.
6. **Answer Key**: The "answer" field in the JSON should be ONLY the letter {correct_option}.
7. **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.

**STYLE MATCHING**: Follow the exact style, complexity, vocabulary, and structure of the provided examples. Match their tone, difficulty level, and presentation format.

{icl_examples}

**RESPONSE FORMAT**: Output ONLY a valid JSON object with this exact structure:

{{
  "topic": "{topic.split('/')[-1]}",
  "question": "...",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "{correct_option}",
  "explanation": "Provide a brief explanation why {correct_option} is correct within 100 words."
}}"""
        
        print(f"📊 [PROMPT] Total prompt length: {len(prompt_template)} characters")
        print(f"🎯 [PROMPT] Correct answer set to: {correct_option}")
        
        return prompt_template, sys_prompt

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
            if len(batch_outs) == 1:
                return batch_outs[0], token_len, generation_time
            else:
                return batch_outs, token_len, generation_time
        else:
            if len(batch_outs) == 1:
                return batch_outs[0]
            else:
                return batch_outs

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
