"""Prompt templates for adversarial distractor strategy"""

from typing import Dict, Tuple

class PromptTemplates:
    
    def get_question_only_prompt(self, topic: str) -> Tuple[str, str]:
        """Get prompt for Phase 1: Question + correct answer generation"""
        
        sys_prompt = """
        You are an expert examiner creating championship-level MCQ questions.
        Generate ONLY the question and correct answer first.
        DO NOT create multiple choice options yet.
        Focus on creating an extremely challenging question that tests deep understanding.
        """
        
        user_prompt = f"""
        Generate an EXTREMELY DIFFICULT question on topic: {topic}

        REQUIREMENTS:
        1. Question must test deep conceptual understanding
        2. Should be championship-level difficulty (only 20-30% can solve)
        3. Provide ONLY the correct answer with detailed reasoning
        4. Include key concepts that could lead to misconceptions

        OUTPUT FORMAT (JSON):
        {{
            "topic": "{topic}",
            "question": "Your challenging question here?",
            "correct_answer": "The factually correct answer",
            "reasoning": "Detailed explanation of why this is correct",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "common_mistakes": ["mistake1", "mistake2"]
        }}
        """
        
        return user_prompt, sys_prompt
    
    def get_misconception_prompt(self, question_data: Dict) -> Tuple[str, str]:
        """Get prompt for misconception-based distractor"""
        
        sys_prompt = """
        You are an expert in student misconceptions and common errors.
        Create a plausible wrong answer based on typical mistakes students make.
        The answer should seem logical but be fundamentally incorrect.
        """
        
        user_prompt = f"""
        QUESTION: {question_data['question']}
        CORRECT ANSWER: {question_data['correct_answer']}
        REASONING: {question_data.get('reasoning', '')}
        COMMON MISTAKES: {question_data.get('common_mistakes', [])}

        Generate a MISCONCEPTION-BASED distractor that:
        1. Represents a common student error or misunderstanding
        2. Seems logical at first glance
        3. Results from applying the wrong concept or method
        4. Would fool someone with partial knowledge

        Examples of misconceptions:
        - Confusing correlation with causation
        - Applying formulas incorrectly
        - Misremembering key facts
        - Using intuition instead of logic

        OUTPUT: Just the distractor answer text (no quotes, no explanation)
        """
        
        return user_prompt, sys_prompt
    
    def get_factual_error_prompt(self, question_data: Dict) -> Tuple[str, str]:
        """Get prompt for factual error distractor"""
        
        sys_prompt = """
        You are an expert at creating subtle but significant factual errors.
        Make answers that are almost correct but have critical mistakes.
        The error should be small but make the answer completely wrong.
        """
        
        user_prompt = f"""
        QUESTION: {question_data['question']}
        CORRECT ANSWER: {question_data['correct_answer']}

        Generate a FACTUAL ERROR distractor by introducing ONE subtle error:

        Error Types:
        1. Wrong numerical value (off by factor, decimal place, sign)
        2. Incorrect unit or measurement
        3. Reversed relationship or causality
        4. Wrong date, time, or sequence
        5. Confused similar concepts or terms

        The error should be:
        - Small enough to seem like a minor mistake
        - Significant enough to make the answer wrong
        - Believable to someone not paying close attention

        OUTPUT: Just the distractor answer text (no quotes, no explanation)
        """
        
        return user_prompt, sys_prompt
    
    def get_semantic_prompt(self, question_data: Dict) -> Tuple[str, str]:
        """Get prompt for semantic similarity distractor"""
        
        sys_prompt = """
        You are an expert at creating answers that sound authoritative but are logically incorrect.
        Focus on using similar vocabulary while ensuring the answer is wrong.
        """
        
        user_prompt = f"""
        QUESTION: {question_data['question']}
        CORRECT ANSWER: {question_data['correct_answer']}
        KEY CONCEPTS: {question_data.get('key_concepts', [])}

        Generate a SEMANTICALLY SIMILAR distractor that:
        1. Uses similar vocabulary and terminology
        2. Sounds authoritative and plausible
        3. Is logically or factually incorrect
        4. Might fool someone who recognizes concepts but doesn't understand deeply

        Techniques:
        - Use correct terminology in wrong context
        - Combine real concepts incorrectly
        - State opposite or inverse relationships
        - Mix up cause and effect

        OUTPUT: Just the distractor answer text (no quotes, no explanation)
        """
        
        return user_prompt, sys_prompt
