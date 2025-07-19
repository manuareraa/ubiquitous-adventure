#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent

import random
import json
import sys
import os

# Add strategy imports with comprehensive verbose logging
print("🔧 [INIT] Loading adversarial distractor strategy...")
try:
    strategy_path = os.path.join(os.path.dirname(__file__), 'q-strategies', 's1-adversarial-distractors')
    print(f"📁 [INIT] Strategy path: {strategy_path}")
    sys.path.append(strategy_path)
    from distractor_engine import AdversarialDistractorEngine
    ADVERSARIAL_STRATEGY_AVAILABLE = True
    print("✅ [INIT] Adversarial distractor strategy loaded successfully!")
    print("🎯 [INIT] Two-phase generation with misconception/factual/semantic distractors enabled")
except ImportError as e:
    ADVERSARIAL_STRATEGY_AVAILABLE = False
    print(f"❌ [INIT] Adversarial distractor strategy not available: {e}")
    print("📝 [INIT] Falling back to standard question generation")

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""
    
    def __init__(self, **kwargs):
        print("🚀 [AGENT] Initializing QuestioningAgent...")
        self.agent = QAgent(**kwargs)
        print(f"🤖 [AGENT] Base QAgent initialized with model type: {self.agent.model_type}")
        
        # Initialize adversarial strategy if available
        if ADVERSARIAL_STRATEGY_AVAILABLE:
            print("🎯 [STRATEGY] Initializing Adversarial Distractor Engine...")
            self.adversarial_engine = AdversarialDistractorEngine(self.agent)
            print("✅ [STRATEGY] Adversarial engine ready - Two-phase generation enabled")
            print("📋 [STRATEGY] Available distractor types: misconception, factual_error, semantic")
        else:
            self.adversarial_engine = None
            print("⚠️  [STRATEGY] Adversarial engine not available - using standard generation only")

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str)->str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            'EXAMPLE: {}\n'
            '{{\n'
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            '}}'
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += fmt.format(topic, topic.split('/')[-1], question, *choices, answer, explanation) + "\n\n"
        return sample_str.strip()

    def build_prompt(self, topic: str, wadvsys: bool = True, wicl: bool = True, inc_samples: List[Dict[str, str]]|None = None) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""
        
        if wadvsys:
            # TODO: Manipulate this SYS prompt for better results
            sys_prompt = """
            You are an **expert-level examiner** with deep expertise in designing **highly challenging and conceptually rigorous multiple-choice questions (MCQs)** for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier competitive exams.
            Think step by step to generate the question and solve the same, but only output the final answer. Do not show your thinking process.
            **Please DO NOT reveal the solution steps or any intermediate reasoning.**
            """
        else:
            sys_prompt = "You are an examiner tasked with creating extremely difficult multiple-choice questions"
            
        tmpl = (
            'Generate an EXTREMELY DIFFICULT MCQ on topic: {0}.\n\n'

            '**CRITICAL REQUIREMENTS:**\n'
            '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
            '2.  **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.\n'
            '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
            '4.  **Single Correct Answer**: Ensure that option {2} is only factually correct.\n'
            '5.  **Plausible Distractors**: While option {3} are three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic, designed to mislead someone without expert knowledge.\n'
            '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
            '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.\n\n'

            '{5}'
            
            'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
            
            'EXAMPLE: {6}\n'
            '{{\n'
            '  "topic": "{7}",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "Provide a brief explanation why {9} is correct within 100 words."\n'
            '}}'
        )
        # Remove model's preferential bias for options
        correct_option = random.choice(['A', 'B', 'C', 'D'])
        distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(topic, topic, correct_option, distractors, correct_option, inc_samples_ex, topic, topic.split('/')[-1], correct_option, correct_option)

        return prompt, sys_prompt


    def generate_question(self, topic: Tuple[str, str]|List[Tuple[str, str]], wadvsys: bool, wicl: bool, inc_samples: Dict[str, List[Dict[str, str]]]|None, **gen_kwargs) -> Tuple[List[str], int|None, float|None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                # Safe ICL lookup with verbose logging
                topic_key = t[1]  # This is the subtopic name
                print(f"🔍 [ICL] Looking for examples for topic: '{topic_key}'")
                
                topic_samples = None
                if inc_samples:
                    if topic_key in inc_samples:
                        topic_samples = inc_samples[topic_key]
                        print(f"✅ [ICL] Found {len(topic_samples)} examples for '{topic_key}'")
                    else:
                        print(f"⚠️  [ICL] No examples found for '{topic_key}' - available keys: {list(inc_samples.keys())}")
                        topic_samples = []
                else:
                    print("⚠️  [ICL] No ICL samples provided")
                    topic_samples = []
                    
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys, wicl, topic_samples)
                prompt.append(p)
        else:
            # Safe ICL lookup for single topic
            topic_key = topic[1]
            print(f"🔍 [ICL] Looking for examples for topic: '{topic_key}'")
            
            topic_samples = None
            if inc_samples:
                if topic_key in inc_samples:
                    topic_samples = inc_samples[topic_key]
                    print(f"✅ [ICL] Found {len(topic_samples)} examples for '{topic_key}'")
                else:
                    print(f"⚠️  [ICL] No examples found for '{topic_key}' - available keys: {list(inc_samples.keys())}")
                    topic_samples = []
            else:
                print("⚠️  [ICL] No ICL samples provided")
                topic_samples = []
                
            prompt, sp = self.build_prompt(f"{topic[0]}/{topic[1]}", wadvsys, wicl, topic_samples)
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt
    
    def generate_adversarial_question(self, topic: str, **kwargs) -> Dict:
        """Generate question using adversarial distractor strategy with comprehensive logging"""
        print(f"\n🎯 [ADVERSARIAL] Starting adversarial question generation for topic: {topic}")
        
        if not self.adversarial_engine:
            print("❌ [ADVERSARIAL] ERROR: Adversarial strategy not available")
            raise ValueError("Adversarial strategy not available")
        
        print("📋 [ADVERSARIAL] Strategy confirmed available - proceeding with two-phase generation")
        
        # Phase 1: Generate question and correct answer
        print("\n🔄 [PHASE-1] Generating question and correct answer only...")
        print(f"📝 [PHASE-1] Using specialized question-only prompts for topic: {topic}")
        
        try:
            question_data = self.adversarial_engine.generate_question_and_answer(topic, **kwargs)
            print("✅ [PHASE-1] Question and correct answer generated successfully")
            print(f"❓ [PHASE-1] Question: {question_data.get('question', 'N/A')[:100]}...")
            print(f"✔️  [PHASE-1] Correct answer: {question_data.get('correct_answer', 'N/A')[:50]}...")
            print(f"🔑 [PHASE-1] Key concepts identified: {question_data.get('key_concepts', [])}")
        except Exception as e:
            print(f"❌ [PHASE-1] ERROR: Failed to generate question data: {e}")
            raise
        
        # Phase 2: Generate adversarial distractors
        print("\n🔄 [PHASE-2] Generating three types of adversarial distractors...")
        print("🧠 [PHASE-2] Type 1: Misconception-based distractors (common student errors)")
        print("⚠️  [PHASE-2] Type 2: Factual error distractors (subtle but critical mistakes)")
        print("🎭 [PHASE-2] Type 3: Semantic similarity distractors (sounds right, logically wrong)")
        
        try:
            distractors = self.adversarial_engine.generate_adversarial_distractors(question_data, **kwargs)
            print("✅ [PHASE-2] All three distractor types generated successfully")
            for i, distractor in enumerate(distractors, 1):
                distractor_type = ["misconception", "factual_error", "semantic"][i-1]
                print(f"🎯 [PHASE-2] Distractor {i} ({distractor_type}): {distractor[:60]}...")
        except Exception as e:
            print(f"❌ [PHASE-2] ERROR: Failed to generate distractors: {e}")
            raise
        
        # Phase 3: Create complete MCQ
        print("\n🔄 [PHASE-3] Assembling final MCQ with randomized choice order...")
        
        try:
            complete_mcq = self.adversarial_engine.create_complete_mcq(question_data, distractors)
            print("✅ [PHASE-3] Complete MCQ assembled successfully")
            print(f"📊 [PHASE-3] Final MCQ structure: {len(complete_mcq.get('choices', []))} choices")
            print(f"🎯 [PHASE-3] Correct answer position: {complete_mcq.get('answer', 'N/A')}")
            print(f"🏷️  [PHASE-3] Strategy metadata: {complete_mcq.get('strategy', 'N/A')}")
            print(f"📋 [PHASE-3] Distractor types: {complete_mcq.get('distractor_types', [])}")
            
            # Validate final MCQ
            if self.adversarial_engine.validator.validate_final_mcq(complete_mcq):
                print("✅ [VALIDATION] MCQ passed all quality checks")
                quality_score = self.adversarial_engine.validator.calculate_quality_score(complete_mcq)
                print(f"📈 [VALIDATION] Quality score: {quality_score:.2f}/1.0")
            else:
                print("⚠️  [VALIDATION] MCQ failed some quality checks but proceeding")
            
            print("🎉 [ADVERSARIAL] Adversarial question generation completed successfully!\n")
            return complete_mcq
            
        except Exception as e:
            print(f"❌ [PHASE-3] ERROR: Failed to create complete MCQ: {e}")
            raise


    def generate_batches(self, num_questions: int, topics: Dict[str, List[str]], batch_size: int = 5, wadvsys: bool=True, wicl: bool = True, inc_samples: Dict[str, List[Dict[str, str]]]|None = None, use_adversarial: bool = False, **kwargs) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        print(f"\n🚀 [BATCH] Starting batch generation: {num_questions} questions")
        print(f"📊 [BATCH] Parameters: batch_size={batch_size}, wadvsys={wadvsys}, wicl={wicl}")
        print(f"🎯 [BATCH] Adversarial strategy requested: {use_adversarial}")
        
        # Check adversarial strategy availability and determine final strategy
        if use_adversarial and not self.adversarial_engine:
            print("⚠️  [BATCH] WARNING: Adversarial strategy requested but not available, using standard generation")
            use_adversarial = False
        elif not use_adversarial and ADVERSARIAL_STRATEGY_AVAILABLE:
            print("🎯 [BATCH] Auto-enabling adversarial strategy (available by default)")
            use_adversarial = True
        
        # Show final strategy decision
        if use_adversarial:
            print("✅ [BATCH] STRATEGY SELECTED: Adversarial Distractor Generation")
            print("🎯 [BATCH] Each question will use two-phase generation with sophisticated distractors")
        else:
            print("📝 [BATCH] STRATEGY SELECTED: Standard Question Generation")
            print("📋 [BATCH] Using traditional batch processing with standard prompts")
        
        extended_topics = self.populate_topics(topics, num_questions)
        print(f"📚 [BATCH] Extended topics generated: {len(extended_topics)} topic combinations")
        
        questions = []
        tls, gts = [], []
        
        if use_adversarial:
            # Use adversarial strategy - process one by one with detailed logging
            print("\n🎯 [ADVERSARIAL-BATCH] Starting adversarial generation mode...")
            print("⚡ [ADVERSARIAL-BATCH] Note: Processing individually for maximum quality (slower but better)")
            pbar = tqdm(total=num_questions, desc="ADVERSARIAL: ")
            
            for i, topic_tuple in enumerate(extended_topics):
                topic_str = f"{topic_tuple[0]}/{topic_tuple[1]}"
                print(f"\n📍 [Q{i+1}/{num_questions}] Processing: {topic_str}")
                
                try:
                    question = self.generate_adversarial_question(topic_str, **kwargs)
                    questions.append(json.dumps(question))
                    tls.append(None)  # Token counting handled internally
                    gts.append(None)  # Timing handled internally
                    print(f"✅ [Q{i+1}] Adversarial question generated successfully")
                    
                except Exception as e:
                    print(f"❌ [Q{i+1}] ERROR in adversarial generation: {e}")
                    print(f"🔄 [Q{i+1}] Falling back to standard generation...")
                    
                    # Fallback to standard generation
                    try:
                        # Ensure we get string output, not tuples
                        fallback_kwargs = kwargs.copy()
                        fallback_kwargs['tgps_show'] = True  # We need timing info
                        
                        fallback_question = self.generate_question([topic_tuple], wadvsys, wicl, inc_samples, **fallback_kwargs)
                        
                        # Extract the actual response string (first element of tuple)
                        response_data = fallback_question[0]
                        if isinstance(response_data, list):
                            questions.extend(response_data)
                        else:
                            questions.append(response_data)
                            
                        tls.append(fallback_question[1])
                        gts.append(fallback_question[2])
                        print(f"✅ [Q{i+1}] Fallback generation successful")
                    except Exception as fallback_error:
                        print(f"❌ [Q{i+1}] CRITICAL: Both adversarial and fallback failed: {fallback_error}")
                        questions.append(json.dumps({"error": "Generation failed", "topic": topic_str}))
                        tls.append(None)
                        gts.append(None)
                
                pbar.update(1)
            
            pbar.close()
            print(f"\n🎉 [ADVERSARIAL-BATCH] Completed! Generated {len(questions)} questions using adversarial strategy")
            
        else:
            # Use standard batch generation with verbose logging
            print("\n📝 [STANDARD-BATCH] Starting standard batch generation mode...")
            total_batches = (len(extended_topics) + batch_size - 1) // batch_size
            print(f"📊 [STANDARD-BATCH] Processing {total_batches} batches of size {batch_size}")
            pbar = tqdm(total=total_batches, desc="STANDARD: ")
            
            for i in range(0, len(extended_topics), batch_size):
                batch_topics = extended_topics[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                print(f"\n📦 [BATCH-{batch_num}] Processing {len(batch_topics)} topics")
                
                try:
                    batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
                    questions.extend(batch_questions[0])
                    tls.append(batch_questions[1])
                    gts.append(batch_questions[2])
                    print(f"✅ [BATCH-{batch_num}] Generated {len(batch_questions[0])} questions successfully")
                except Exception as e:
                    print(f"❌ [BATCH-{batch_num}] ERROR: {e}")
                    # Add empty results for failed batch
                    questions.extend([''] * len(batch_topics))
                    tls.append(None)
                    gts.append(None)
                
                pbar.update(1)
            
            # Handle remaining topics if any
            if len(extended_topics) % batch_size != 0:
                remaining_topics = extended_topics[-(len(extended_topics) % batch_size):]
                print(f"\n📦 [FINAL-BATCH] Processing remaining {len(remaining_topics)} topics")
                
                try:
                    batch_questions = self.generate_question(remaining_topics, wadvsys, wicl, inc_samples, **kwargs)
                    questions.extend(batch_questions[0])
                    tls.append(batch_questions[1])
                    gts.append(batch_questions[2])
                    print(f"✅ [FINAL-BATCH] Generated {len(batch_questions[0])} questions successfully")
                except Exception as e:
                    print(f"❌ [FINAL-BATCH] ERROR: {e}")
                    questions.extend([''] * len(remaining_topics))
                    tls.append(None)
                    gts.append(None)
                
                pbar.update(1)
            
            pbar.close()
            print(f"\n🎉 [STANDARD-BATCH] Completed! Generated {len(questions)} questions using standard strategy")
        
        # Final summary
        successful_questions = sum(1 for q in questions if q and q != '' and 'error' not in q.lower())
        print(f"\n📈 [SUMMARY] Total questions generated: {len(questions)}")
        print(f"✅ [SUMMARY] Successful generations: {successful_questions}")
        print(f"❌ [SUMMARY] Failed generations: {len(questions) - successful_questions}")
        print(f"🎯 [SUMMARY] Strategy used: {'Adversarial' if use_adversarial else 'Standard'}")
        
        # JSON Validation and Self-Correction (based on original file approach)
        print(f"\n🔍 [VALIDATION] Starting JSON validation and correction...")
        validated_questions = []
        
        for i, q in enumerate(questions):
            if not q:  # Skip empty questions
                print(f"❌ [Q{i+1}] Empty question, skipping")
                continue
                
            try:
                # Try to parse as JSON first
                if isinstance(q, str):
                    json.loads(q)
                    validated_questions.append(q)
                    print(f"✅ [Q{i+1}] Valid JSON detected")
                elif isinstance(q, dict):
                    validated_questions.append(q)
                    print(f"✅ [Q{i+1}] Valid dict object detected")
            except json.JSONDecodeError as e:
                print(f"❌ [Q{i+1}] Invalid JSON format: {str(e)[:100]}...")
                print(f"🔧 [Q{i+1}] Attempting self-correction...")
                
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
                
                try:
                    # Generate corrected JSON without thinking mode
                    corrected_q = self.agent.generate_response(
                        correction_prompt.format(q),
                        "You are an expert JSON extractor.",
                        enable_thinking=False,
                        thinking_stage="json_correction",
                        max_new_tokens=1024,
                        temperature=0.0,
                        do_sample=False
                    )
                    
                    # Validate corrected output
                    json.loads(corrected_q)
                    validated_questions.append(corrected_q)
                    print(f"✅ [Q{i+1}] Self-correction successful")
                    
                except (json.JSONDecodeError, Exception) as correction_error:
                    print(f"❌ [Q{i+1}] Self-correction failed: {str(correction_error)[:50]}...")
                    print(f"🔄 [Q{i+1}] Skipping malformed question")
                    continue
        
        print(f"\n📊 [VALIDATION] {len(validated_questions)}/{len(questions)} questions passed JSON validation")
        
        # Apply MCQ structure filtering
        print(f"🔍 [FILTERING] Applying MCQ structure validation...")
        filtered_questions = self.filter_questions(validated_questions)
        
        print(f"📈 [FINAL] {len(filtered_questions)} questions ready for output")
        
        return filtered_questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(self, questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
        """Filter and validate questions to ensure proper JSON format and MCQ structure"""
        def basic_checks(q2: Dict[str, str])->bool:
            # check required keys
            required_keys = ['topic', 'question', 'choices', 'answer']
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
                if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                    # check answer format
                    # Check token length
                    check_len = sum(self.count_tokens_q(q2[k]) for k in ['question', 'answer'])
                    check_len += sum(self.count_tokens_q(choice) for choice in q2['choices']) - 15
                    if check_len < 130:
                        if check_len + self.count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2['answer'], str):
                                return True
            return False
            
        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
                    print(f"✅ [FILTER] Question {i+1} passed validation")
                else:
                    print(f"❌ [FILTER] Question {i+1} failed validation (dict format)")
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                        print(f"✅ [FILTER] Question {i+1} passed validation")
                    else:
                        print(f"❌ [FILTER] Question {i+1} failed validation (parsed JSON)")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"❌ [FILTER] Skipping invalid JSON at index {i}: {q[:100]}...")
                    continue
            else:
                print(f"❌ [FILTER] Question {i+1} has invalid type: {type(q)}")
                continue
                
        print(f"📊 [FILTER] {len(correct_format_question)}/{len(questions)} questions passed validation")
        
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        else:
            print(f"⚠️ [FILTER] Too many failed questions ({len(correct_format_question)}/{len(questions)}), returning empty list")
            return list()
    
    def save_questions(self, questions: Any, file_path: str|Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(questions, f, indent=4)
    
    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        
        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str|Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples

# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    argparser.add_argument("--num_questions", type=int, default=200, help="Total number of questions to generate.")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file name to save the generated questions.")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for generating questions.")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = argparser.parse_args()

    print("\n" + "="*80)
    print("🚀 [CLI] Question Agent Starting - Enhanced with Adversarial Strategy")
    print("="*80)
    
    print(f"📊 [CLI] Configuration: {args.num_questions} questions, batch_size={args.batch_size}")
    print(f"📁 [CLI] Output file: {args.output_file}")
    print(f"🔍 [CLI] Verbose mode: {args.verbose}")
    
    # Load configuration files with verbose logging
    print("\n📂 [CONFIG] Loading configuration files...")
    
    try:
        inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")
        print("✅ [CONFIG] In-context learning samples loaded successfully")
        print(f"📋 [CONFIG] ICL samples available for: {list(inc_samples.keys())}")
    except Exception as e:
        print(f"❌ [CONFIG] ERROR loading ICL samples: {e}")
        inc_samples = {}

    try:
        with open("assets/topics.json") as f: 
            topics = json.load(f)
        print("✅ [CONFIG] Topics file loaded successfully")
        total_subtopics = sum(len(subtopics) for subtopics in topics.values())
        print(f"📚 [CONFIG] Available topics: {list(topics.keys())} ({total_subtopics} total subtopics)")
    except Exception as e:
        print(f"❌ [CONFIG] ERROR loading topics: {e}")
        topics = {"General": ["Mathematics", "Science"]}
    
    # Initialize agent with verbose logging
    print("\n🤖 [AGENT] Initializing QuestioningAgent...")
    agent = QuestioningAgent()
    
    # Load generation parameters
    print("\n⚙️  [CONFIG] Loading generation parameters...")
    gen_kwargs = {"tgps_show": True}
    
    try:
        with open("qgen.yaml", "r") as f: 
            yaml_config = yaml.safe_load(f)
            gen_kwargs.update(yaml_config)
        print("✅ [CONFIG] YAML configuration loaded successfully")
        print(f"📊 [CONFIG] Generation parameters: {yaml_config}")
    except Exception as e:
        print(f"⚠️  [CONFIG] WARNING: Could not load qgen.yaml: {e}")
        print("📝 [CONFIG] Using default generation parameters")
    
    # Determine and display strategy selection
    print("\n🎯 [STRATEGY] Determining generation strategy...")
    use_adversarial = ADVERSARIAL_STRATEGY_AVAILABLE
    
    if use_adversarial:
        print("✅ [STRATEGY] ADVERSARIAL STRATEGY ENABLED BY DEFAULT")
        print("🎯 [STRATEGY] Two-phase generation with sophisticated distractors will be used")
        print("📋 [STRATEGY] Features: misconception/factual/semantic distractors, quality validation")
    else:
        print("📝 [STRATEGY] STANDARD STRATEGY (Adversarial not available)")
        print("📋 [STRATEGY] Traditional batch generation will be used")
    
    # Start generation with comprehensive logging
    print("\n" + "-"*80)
    print("🚀 [GENERATION] Starting question generation process...")
    print("-"*80)
    
    questions, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics, 
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        use_adversarial=use_adversarial,
        **gen_kwargs
    )
    print(f"Generated {len(questions)} questions!")
    if args.verbose:
        for q in questions:
            print(q, flush=True)
        print("\n" + "="*50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            # Filter None values to prevent TypeError
            valid_gts = [t for t in gts if t is not None]
            valid_tls = [t for t in tls if t is not None]
            if valid_gts and valid_tls:
                print("Time taken per batch generation:", valid_gts)
                print("Tokens generated per batch:", valid_tls)
                print(f"Total Time Taken: {sum(valid_gts):.3f} seconds; Total Tokens: {sum(valid_tls)}; TGPS: {sum(valid_tls)/sum(valid_gts):.3f} seconds\n\n")
            else:
                print("⚠️ [CLI] No valid timing data available")
        print("\n" + "+"*50 + "\n")

    # Save questions (validation and filtering already done in generate_batches)
    if questions:
        agent.save_questions(questions, args.output_file)
        print(f"✅ [SAVE] Saved {len(questions)} validated questions to {args.output_file}")
    else:
        print(f"❌ [ERROR] No valid questions generated! Check your prompts and model configuration.")
        # Still save empty file for debugging
        agent.save_questions([], args.output_file.replace('.json', '_failed.json'))

    # ========================================================================================
