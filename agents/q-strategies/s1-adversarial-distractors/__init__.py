"""
Adversarial Distractor Strategy (S1)
Two-phase MCQ generation with sophisticated distractor construction
"""

from .distractor_engine import AdversarialDistractorEngine
from .prompt_templates import PromptTemplates
from .validator import DistractorValidator
from .config import StrategyConfig

__all__ = [
    'AdversarialDistractorEngine',
    'PromptTemplates', 
    'DistractorValidator',
    'StrategyConfig'
]
