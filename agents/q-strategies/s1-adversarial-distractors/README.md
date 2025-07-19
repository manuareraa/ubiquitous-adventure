# Adversarial Distractor Strategy (S1)

## Overview
The Adversarial Distractor Strategy is a sophisticated two-phase MCQ generation approach that creates highly challenging questions with strategically crafted wrong answers designed to mislead even knowledgeable test-takers.

## Strategy Components

### 1. **Two-Phase Generation Process**

#### Phase 1: Question & Correct Answer Generation
- Generates the core question with only the correct answer
- Focuses on creating championship-level difficulty questions
- Includes reasoning and key concepts for distractor generation

#### Phase 2: Adversarial Distractor Generation
- Creates three types of sophisticated wrong answers:
  1. **Misconception-based**: Based on common student errors
  2. **Factual error**: Subtle but critical mistakes
  3. **Semantic similarity**: Sounds right but logically wrong

### 2. **File Structure**
```
s1-adversarial-distractors/
├── __init__.py              # Module initialization
├── config.py                # Strategy configuration
├── distractor_engine.py     # Core generation logic
├── prompt_templates.py      # Specialized prompts
├── validator.py             # Quality validation
└── README.md               # This documentation
```

## Core Classes

### `AdversarialDistractorEngine`
Main orchestrator for the two-phase generation process.

**Key Methods:**
- `generate_question_and_answer(topic)`: Phase 1 generation
- `generate_adversarial_distractors(question_data)`: Phase 2 generation
- `create_complete_mcq(question_data, distractors)`: Final assembly

### `PromptTemplates`
Specialized prompts for each distractor type.

**Prompt Types:**
- `get_question_only_prompt()`: Phase 1 question generation
- `get_misconception_prompt()`: Misconception-based distractors
- `get_factual_error_prompt()`: Factual error distractors
- `get_semantic_prompt()`: Semantic similarity distractors

### `DistractorValidator`
Quality assurance for generated content.

**Validation Types:**
- Question data structure validation
- Distractor uniqueness and quality
- Final MCQ format compliance
- Quality scoring metrics

### `StrategyConfig`
Configuration parameters for the strategy.

**Key Settings:**
- Token limits for each phase
- Temperature settings for creativity
- Quality thresholds
- Distractor type weights

## Usage Examples

### 1. **Command Line Usage**

```bash
# Standard generation (existing behavior)
python -m agents.question_agent --num_questions 10 --strategy standard

# Adversarial generation (new strategy)
python -m agents.question_agent --num_questions 10 --strategy adversarial --verbose

# Custom batch size with adversarial strategy
python -m agents.question_agent --num_questions 20 --strategy adversarial --batch_size 1
```

### 2. **Programmatic Usage**

```python
from agents.question_agent import QuestioningAgent

# Initialize agent
agent = QuestioningAgent()

# Generate single adversarial question
question = agent.generate_adversarial_question("Algebra/Linear Equations")

# Generate batch with adversarial strategy
questions, tls, gts = agent.generate_batches(
    num_questions=5,
    topics={"Mathematics": ["Algebra", "Geometry"]},
    use_adversarial=True
)
```

## Configuration

### Strategy Parameters (config.py)
```python
QUESTION_PHASE_TOKENS = 800      # Tokens for question generation
DISTRACTOR_PHASE_TOKENS = 300    # Tokens per distractor
DISTRACTOR_TEMPERATURE = 0.8     # Creativity for distractors

# Distractor type weights
MISCONCEPTION_WEIGHT = 0.4
FACTUAL_ERROR_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.3
```

### YAML Configuration
The strategy respects existing `qgen.yaml` settings and adds strategy-specific parameters.

## Quality Metrics

### Distractor Quality Indicators
1. **Plausibility**: How believable wrong answers are
2. **Diversity**: Variation between distractor types
3. **Difficulty**: Challenge level for test-takers
4. **Fairness**: Absence of ambiguity or tricks

### Validation Checks
- No distractor matches correct answer
- All distractors are unique
- Minimum content length requirements
- Proper JSON structure compliance

## Expected Improvements

### Quality Enhancements
- **50-70% increase** in question difficulty
- **More plausible** wrong answers
- **Better discrimination** between knowledge levels
- **Reduced guessing success rate**

### Measurable Outcomes
- More evenly distributed distractor selection
- Higher expert evaluation scores
- Better student performance differentiation
- Increased time spent per question

## Integration Points

### With question_agent.py
- New `generate_adversarial_question()` method
- Enhanced `generate_batches()` with strategy support
- Command line `--strategy` argument

### With question_model.py
- Strategy mode support in `QAgent`
- `generate_response_with_strategy()` method
- Strategy parameter merging

## Error Handling

### Fallback Mechanisms
1. **Import Failure**: Graceful degradation to standard generation
2. **Generation Errors**: Automatic fallback with error logging
3. **Validation Failures**: Default distractor creation
4. **JSON Parsing**: Structured fallback responses

### Debugging Features
- Verbose error messages
- Strategy availability warnings
- Quality validation feedback
- Progress tracking with strategy indicators

## Best Practices

### For Optimal Results
1. **Start Small**: Test with 1-5 questions initially
2. **Monitor Quality**: Check generated questions manually
3. **Adjust Parameters**: Tune temperature and token limits
4. **Validate Topics**: Ensure topic files are properly formatted
5. **Review Outputs**: Examine distractor quality regularly

### Performance Considerations
- Adversarial generation is slower (2-3x) than standard
- Memory usage increases with longer prompts
- Consider batch size reduction for adversarial mode
- Monitor token consumption for cost management

## Troubleshooting

### Common Issues
1. **Strategy Not Available**: Check import paths and file structure
2. **Poor Distractor Quality**: Adjust temperature and prompt templates
3. **JSON Parsing Errors**: Verify model output format
4. **Memory Issues**: Reduce batch size or token limits

### Debug Commands
```bash
# Test strategy availability
python3 -c "import sys; sys.path.append('agents/q-strategies/s1-adversarial-distractors'); from distractor_engine import AdversarialDistractorEngine; print('✅ Success')"

# Verbose generation
python -m agents.question_agent --num_questions 1 --strategy adversarial --verbose
```

## Future Enhancements

### Planned Features
1. **A/B Testing Framework**: Compare strategy effectiveness
2. **Human Evaluation Interface**: Collect quality feedback
3. **Adaptive Difficulty**: Adjust based on performance
4. **Multi-Strategy Support**: Additional generation strategies
5. **Quality Metrics Dashboard**: Real-time quality monitoring

### Extension Points
- Custom distractor types
- Domain-specific prompt templates
- Advanced validation rules
- Performance optimization
- Integration with evaluation systems

This strategy represents a significant advancement in automated MCQ generation, focusing on creating questions that truly challenge and assess deep understanding rather than surface-level knowledge.
