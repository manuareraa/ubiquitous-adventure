"""Quality validation for adversarial distractors"""

import json
from typing import Dict, List
from config import StrategyConfig

class DistractorValidator:
    
    def __init__(self):
        self.config = StrategyConfig()
    
    def validate_question_data(self, question_data: Dict) -> bool:
        """Validate Phase 1 question data"""
        required_fields = ['topic', 'question', 'correct_answer', 'reasoning']
        
        if not all(field in question_data for field in required_fields):
            return False
        
        # Check for minimum content length
        if len(question_data['question']) < 20:
            return False
        
        if len(question_data['correct_answer']) < 5:
            return False
        
        return True
    
    def validate_distractors(self, distractors: List[str], correct_answer: str) -> bool:
        """Validate generated distractors"""
        if len(distractors) != 3:
            return False
        
        # Check no distractor is identical to correct answer
        for distractor in distractors:
            if distractor.strip().lower() == correct_answer.strip().lower():
                return False
        
        # Check distractors are not identical to each other
        unique_distractors = set(d.strip().lower() for d in distractors)
        if len(unique_distractors) != 3:
            return False
        
        # Check minimum length
        for distractor in distractors:
            if len(distractor.strip()) < 3:
                return False
        
        return True
    
    def validate_final_mcq(self, mcq_data: Dict) -> bool:
        """Validate final MCQ format"""
        required_fields = ['topic', 'question', 'choices', 'answer', 'explanation']
        
        if not all(field in mcq_data for field in required_fields):
            return False
        
        # Check choices format
        if len(mcq_data['choices']) != 4:
            return False
        
        # Check answer is valid letter
        if mcq_data['answer'] not in ['A', 'B', 'C', 'D']:
            return False
        
        return True
    
    def calculate_quality_score(self, mcq_data: Dict) -> float:
        """Calculate overall quality score for the MCQ"""
        score = 0.0
        
        # Question complexity (0-0.3)
        question_length = len(mcq_data['question'])
        if question_length > 100:
            score += 0.3
        elif question_length > 50:
            score += 0.2
        else:
            score += 0.1
        
        # Choice diversity (0-0.3)
        choices_text = [choice.split(') ', 1)[1] for choice in mcq_data['choices']]
        avg_length = sum(len(choice) for choice in choices_text) / 4
        if avg_length > 20:
            score += 0.3
        elif avg_length > 10:
            score += 0.2
        else:
            score += 0.1
        
        # Explanation quality (0-0.2)
        explanation_length = len(mcq_data.get('explanation', ''))
        if explanation_length > 50:
            score += 0.2
        elif explanation_length > 20:
            score += 0.1
        
        # Strategy completion (0-0.2)
        if mcq_data.get('strategy') == self.config.STRATEGY_NAME:
            score += 0.2
        
        return min(score, 1.0)
