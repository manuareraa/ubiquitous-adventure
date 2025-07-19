"""Core engine for adversarial distractor generation"""

import json
import random
from typing import Dict, List, Tuple
from .prompt_templates import PromptTemplates
from .config import StrategyConfig
from .validator import DistractorValidator

class AdversarialDistractorEngine:
    def __init__(self, model_agent):
        self.model_agent = model_agent
        self.prompts = PromptTemplates()
        self.config = StrategyConfig()
        self.validator = DistractorValidator()
    
    def generate_question_and_answer(self, topic: str, **kwargs) -> Dict:
        """Phase 1: Generate question with correct answer only"""
        prompt, sys_prompt = self.prompts.get_question_only_prompt(topic)
        
        response = self.model_agent.generate_response(
            prompt, 
            sys_prompt, 
            max_new_tokens=self.config.QUESTION_PHASE_TOKENS,
            **kwargs
        )
        
        try:
            question_data = json.loads(response)
            if not self.validator.validate_question_data(question_data):
                raise ValueError("Invalid question data format")
            return question_data
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: create basic structure
            return {
                "topic": topic,
                "question": f"Generate a challenging question about {topic}",
                "correct_answer": "Correct answer placeholder",
                "reasoning": "Reasoning placeholder",
                "key_concepts": [topic],
                "common_mistakes": ["common mistake"]
            }
    
    def generate_adversarial_distractors(self, question_data: Dict, **kwargs) -> List[str]:
        """Phase 2: Generate three types of adversarial distractors"""
        distractors = []
        
        try:
            # Type 1: Misconception-based
            misconception = self._generate_misconception_distractor(question_data, **kwargs)
            distractors.append(misconception)
            
            # Type 2: Factual error
            factual_error = self._generate_factual_error_distractor(question_data, **kwargs)
            distractors.append(factual_error)
            
            # Type 3: Semantic similarity
            semantic = self._generate_semantic_distractor(question_data, **kwargs)
            distractors.append(semantic)
            
            # Validate distractors
            if not self.validator.validate_distractors(distractors, question_data['correct_answer']):
                # Fallback distractors
                distractors = [
                    f"Incorrect option based on {question_data.get('topic', 'topic')} misconception",
                    f"Wrong answer with factual error related to {question_data.get('topic', 'topic')}",
                    f"Semantically similar but incorrect answer for {question_data.get('topic', 'topic')}"
                ]
        
        except Exception as e:
            print(f"Warning: Error generating distractors: {e}")
            # Fallback distractors
            distractors = [
                "Fallback distractor option A",
                "Fallback distractor option B", 
                "Fallback distractor option C"
            ]
        
        return distractors
    
    def _generate_misconception_distractor(self, question_data: Dict, **kwargs) -> str:
        """Generate misconception-based distractor"""
        prompt, sys_prompt = self.prompts.get_misconception_prompt(question_data)
        
        response = self.model_agent.generate_response(
            prompt, 
            sys_prompt,
            max_new_tokens=self.config.DISTRACTOR_PHASE_TOKENS,
            temperature=self.config.DISTRACTOR_TEMPERATURE,
            **kwargs
        )
        
        return response.strip()
    
    def _generate_factual_error_distractor(self, question_data: Dict, **kwargs) -> str:
        """Generate factual error distractor"""
        prompt, sys_prompt = self.prompts.get_factual_error_prompt(question_data)
        
        response = self.model_agent.generate_response(
            prompt, 
            sys_prompt,
            max_new_tokens=self.config.DISTRACTOR_PHASE_TOKENS,
            temperature=self.config.DISTRACTOR_TEMPERATURE,
            **kwargs
        )
        
        return response.strip()
    
    def _generate_semantic_distractor(self, question_data: Dict, **kwargs) -> str:
        """Generate semantic similarity distractor"""
        prompt, sys_prompt = self.prompts.get_semantic_prompt(question_data)
        
        response = self.model_agent.generate_response(
            prompt, 
            sys_prompt,
            max_new_tokens=self.config.DISTRACTOR_PHASE_TOKENS,
            temperature=self.config.DISTRACTOR_TEMPERATURE,
            **kwargs
        )
        
        return response.strip()
    
    def create_complete_mcq(self, question_data: Dict, distractors: List[str]) -> Dict:
        """Combine question and distractors into final MCQ format"""
        # Combine correct answer with distractors
        all_choices = [question_data['correct_answer']] + distractors
        
        # Randomize order
        random.shuffle(all_choices)
        
        # Find correct answer position
        correct_position = all_choices.index(question_data['correct_answer'])
        correct_letter = ['A', 'B', 'C', 'D'][correct_position]
        
        # Format with letters
        formatted_choices = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(all_choices)]
        
        mcq_data = {
            "topic": question_data['topic'],
            "question": question_data['question'],
            "choices": formatted_choices,
            "answer": correct_letter,
            "explanation": question_data.get('reasoning', ''),
            "strategy": self.config.STRATEGY_NAME,
            "distractor_types": ["misconception", "factual_error", "semantic"]
        }
        
        # Validate final MCQ
        if not self.validator.validate_final_mcq(mcq_data):
            print("Warning: Generated MCQ failed validation")
        
        return mcq_data
