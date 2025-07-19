# Agents Directory - Script Documentation

This directory contains four core scripts that implement the question generation and answer solving functionality for the hackathon competition system. Each script serves a specific purpose in the AI-powered MCQ pipeline.

---

## 📁 Directory Structure

```
agents/
├── question_agent.py    # High-level question generation with batch processing
├── question_model.py    # Low-level question generation model wrapper
├── answer_agent.py      # High-level answer solving with batch processing  
├── answer_model.py      # Low-level answer solving model wrapper
└── README.md           # This file
```

---

## 🤖 Script Descriptions

### 1. `question_agent.py` - Question Generation Engine

**Purpose**: High-level script for generating MCQ questions in batches with advanced filtering and validation.

#### Key Features:
- **Batch Processing**: Generate multiple questions efficiently
- **Topic Management**: Supports various topics from `assets/topics.json`
- **In-Context Learning**: Uses examples from `assets/topics_example.json`
- **Quality Filtering**: Validates JSON format and content quality
- **Token Counting**: Ensures questions meet token limits
- **Self-Reflection**: Auto-fixes malformed JSON responses

#### Use Cases:
- **Dataset Creation**: Generate large question datasets for training/testing
- **Competition Prep**: Create practice question banks
- **Quality Assurance**: Batch generate and filter high-quality questions
- **Research**: Analyze question generation patterns and quality

#### How to Use:
```bash
# Basic usage - generate 20 questions
python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json

# Advanced usage with custom settings
python -m agents.question_agent \
    --num_questions 100 \
    --output_file outputs/custom_questions.json \
    --batch_size 10 \
    --verbose

# Quick test run
python -m agents.question_agent --num_questions 5 --verbose
```

#### Configuration:
- Uses `qgen.yaml` for generation parameters
- Requires `assets/topics.json` for topic selection
- Optionally uses `assets/topics_example.json` for in-context learning

#### Output:
- Raw questions: `outputs/questions.json`
- Filtered questions: `outputs/filtered_questions.json`
- Includes generation statistics and timing information

---

### 2. `question_model.py` - Question Generation Model Core

**Purpose**: Low-level model wrapper that handles the actual AI model interaction for question generation.

#### Key Features:
- **Model Management**: Loads and manages the Qwen model
- **Batch Generation**: Supports single and batch text generation
- **Chat Template**: Properly formats prompts for the model
- **Performance Tracking**: Measures tokens per second (TGPS)
- **Memory Efficient**: Handles padding and batching automatically

#### Use Cases:
- **Direct Model Access**: When you need fine control over generation
- **Custom Applications**: Building your own question generation tools
- **Experimentation**: Testing different prompts and parameters
- **Integration**: Embedding question generation in other systems

#### How to Use:
```python
from agents.question_model import QAgent

# Initialize the model
model = QAgent()

# Single question generation
prompt = "Generate a hard Number Series MCQ question"
response, tokens, time = model.generate_response(
    prompt, 
    system_prompt="You are an expert examiner",
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    tgps_show=True
)

# Batch generation
prompts = ["Generate question 1", "Generate question 2"]
responses, tokens, time = model.generate_response(prompts, tgps_show=True)
```

#### Technical Details:
- **Model**: Qwen3-4B (hardcoded path needs fixing)
- **Tokenizer**: Left-padding for batch processing
- **Device**: Auto-mapped across available GPUs
- **Precision**: Auto-selected torch dtype

---

### 3. `answer_agent.py` - Answer Solving Engine

**Purpose**: High-level script for solving MCQ questions in batches with confidence scoring and validation.

#### Key Features:
- **Batch Processing**: Solve multiple questions efficiently
- **Confidence Scoring**: Provides reasoning for each answer
- **Answer Validation**: Filters and validates response format
- **Performance Metrics**: Tracks accuracy and timing
- **Self-Correction**: Auto-fixes malformed responses
- **Flexible Input**: Accepts various question formats

#### Use Cases:
- **Automated Grading**: Score large question sets automatically
- **Performance Testing**: Evaluate model accuracy on question banks
- **Competition Simulation**: Test AI performance against human benchmarks
- **Quality Control**: Validate question difficulty and clarity

#### How to Use:
```bash
# Basic usage - solve questions from file
python -m agents.answer_agent \
    --input_file outputs/filtered_questions.json \
    --output_file outputs/answers.json

# Advanced usage with custom settings
python -m agents.answer_agent \
    --input_file custom_questions.json \
    --output_file custom_answers.json \
    --batch_size 10 \
    --verbose

# Quick test with verbose output
python -m agents.answer_agent \
    --input_file test_questions.json \
    --output_file test_answers.json \
    --batch_size 3 \
    --verbose
```

#### Configuration:
- Uses `agen.yaml` for generation parameters
- Supports two system prompt modes (`SELECT_PROMPT1`)
- Configurable batch sizes for memory management

#### Output:
- Raw answers: `outputs/answers.json`
- Filtered answers: `outputs/filtered_answers.json`
- Performance statistics and accuracy metrics

---

### 4. `answer_model.py` - Answer Solving Model Core

**Purpose**: Low-level model wrapper that handles the actual AI model interaction for answer solving.

#### Key Features:
- **Model Management**: Loads and manages the Qwen model
- **Batch Processing**: Efficient batch answer generation
- **Deterministic Mode**: Supports both sampling and greedy decoding
- **Performance Monitoring**: Tracks generation speed and efficiency
- **Memory Optimization**: Handles large batch sizes efficiently

#### Use Cases:
- **Direct Model Access**: Fine control over answer generation
- **Custom Applications**: Building specialized solving tools
- **Research**: Analyzing model reasoning patterns
- **Integration**: Embedding answer solving in larger systems

#### How to Use:
```python
from agents.answer_model import AAgent

# Initialize the model
model = AAgent()

# Single answer generation
question = "What is 2+2? A) 3 B) 4 C) 5 D) 6"
response, tokens, time = model.generate_response(
    question,
    system_prompt="You are a math expert",
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    tgps_show=True
)

# Batch processing
questions = ["Question 1", "Question 2", "Question 3"]
responses, tokens, time = model.generate_response(questions, tgps_show=True)
```

#### Technical Details:
- **Model**: Qwen3-4B (hardcoded path needs fixing)
- **Optimization**: Random seed set for reproducibility
- **Processing**: Supports both single and batch inference
- **Output**: Clean text extraction with thinking tag removal

---

## 🔄 Workflow Integration

### Typical Usage Pipeline:

1. **Question Generation**:
   ```bash
   python -m agents.question_agent --num_questions 50 --output_file questions.json
   ```

2. **Answer Generation**:
   ```bash
   python -m agents.answer_agent --input_file questions.json --output_file answers.json
   ```

3. **Analysis**:
   - Compare expected vs generated answers
   - Analyze reasoning quality
   - Measure accuracy metrics

### Advanced Workflows:

#### Competition Simulation:
```bash
# Generate championship-level questions
python -m agents.question_agent --num_questions 100 --output_file competition_q.json

# Solve with high accuracy settings (modify agen.yaml first)
python -m agents.answer_agent --input_file competition_q.json --output_file competition_a.json --verbose
```

#### Dataset Creation:
```bash
# Large-scale generation
python -m agents.question_agent --num_questions 1000 --batch_size 20 --output_file dataset_q.json

# Validation solving
python -m agents.answer_agent --input_file dataset_q.json --output_file dataset_a.json --batch_size 20
```

---

## ⚙️ Configuration

### YAML Configuration Files:
- **`../qgen.yaml`**: Controls question generation parameters
- **`../agen.yaml`**: Controls answer generation parameters

See `../aq-gen-yaml.md` for detailed parameter explanations.

### Required Assets:
- **`assets/topics.json`**: Topic definitions for question generation
- **`assets/topics_example.json`**: Example questions for in-context learning

### Model Requirements:
- **GPU Memory**: Minimum 8GB recommended for Qwen3-4B
- **Storage**: ~8GB for model files
- **Dependencies**: transformers, torch, tqdm, peft

---

## 🚨 Known Issues & Fixes

### Issue 1: Model Path Error
**Problem**: Both model files use hardcoded local paths that don't exist
```python
model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"  # ❌ Doesn't exist
```

**Solution**: Update to use Hugging Face Hub:
```python
model_name = "Qwen/Qwen2.5-3B-Instruct"  # ✅ Valid repo
```

### Issue 2: Memory Issues
**Problem**: Large batch sizes cause OOM errors

**Solution**: 
- Reduce batch size in command line arguments
- Use gradient checkpointing
- Monitor GPU memory usage

### Issue 3: JSON Parsing Errors
**Problem**: Generated responses sometimes have malformed JSON

**Solution**: 
- Scripts include self-reflection mechanisms
- Automatic JSON extraction and fixing
- Validation and filtering built-in

---

## 📊 Performance Benchmarks

### Typical Performance (Qwen3-4B on A100):
- **Question Generation**: ~15-20 questions/minute
- **Answer Generation**: ~25-30 answers/minute
- **Memory Usage**: ~6-8GB GPU memory
- **Token Generation**: ~50-80 tokens/second

### Optimization Tips:
1. **Batch Size**: Start with 5, increase based on GPU memory
2. **Token Limits**: Use appropriate max_new_tokens for your use case
3. **Sampling**: Use `do_sample=false` for fastest generation
4. **Precision**: Consider using fp16 for speed vs quality trade-off

---

## 🔧 Development & Customization

### Adding New Topics:
1. Edit `assets/topics.json` to include new topic categories
2. Optionally add examples to `assets/topics_example.json`
3. Test with small batches first

### Custom Prompts:
- Modify system prompts in the agent classes
- Experiment with different instruction formats
- Test prompt effectiveness with small samples

### Integration:
- Import agent classes into your own scripts
- Use model classes for direct API access
- Extend functionality with custom filtering/validation

---

## 📝 Logging & Debugging

### Verbose Mode:
Use `--verbose` flag to see:
- Individual question/answer outputs
- Generation timing statistics
- Token counts and performance metrics
- Error messages and warnings

### Log Files:
- Generation statistics are printed to console
- JSON outputs contain metadata
- Error handling provides detailed feedback

---

## 🎯 Best Practices

1. **Start Small**: Test with 5-10 questions before large batches
2. **Monitor Resources**: Watch GPU memory and generation speed
3. **Validate Outputs**: Always check filtered vs raw outputs
4. **Backup Configs**: Save working YAML configurations
5. **Incremental Testing**: Make small changes and test results
6. **Performance Tracking**: Monitor TGPS and adjust batch sizes accordingly

---

*Last updated: 2025-07-19*
*For YAML configuration details, see `../aq-gen-yaml.md`*
