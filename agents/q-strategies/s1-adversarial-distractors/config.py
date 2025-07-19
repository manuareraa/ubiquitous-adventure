"""Configuration for Adversarial Distractor Strategy"""

class StrategyConfig:
    # Generation parameters
    QUESTION_PHASE_TOKENS = 800
    DISTRACTOR_PHASE_TOKENS = 300
    DISTRACTOR_TEMPERATURE = 0.8
    
    # Distractor weights
    MISCONCEPTION_WEIGHT = 0.4
    FACTUAL_ERROR_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.3
    
    # Quality thresholds
    MIN_PLAUSIBILITY_SCORE = 0.7
    MIN_DIVERSITY_SCORE = 0.6
    
    # Strategy metadata
    STRATEGY_NAME = "adversarial-distractors"
    STRATEGY_VERSION = "1.0"
    DESCRIPTION = "Two-phase generation with adversarial distractors"
