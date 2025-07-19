# YAML Configuration Guide for Answer & Question Generation

This document explains the YAML configuration files used to control the behavior of the answer and question generation agents in this hackathon project.

## Overview

The project uses two YAML configuration files:
- **`agen.yaml`** - Controls Answer Agent behavior
- **`qgen.yaml`** - Controls Question Agent behavior

These files contain generation parameters that are loaded at runtime by the respective agent scripts in the `/agents` directory.

---

## 📄 `agen.yaml` - Answer Agent Configuration

**Purpose**: Controls how the Answer Agent generates responses when solving MCQ questions.

### Current Configuration:
```yaml
max_new_tokens: 512
temperature: 0.1
top_p: 0.9
repetition_penalty: 1.2
do_sample: true
```

### Parameter Details:

#### `max_new_tokens: 512`
- **What it does**: Maximum number of new tokens the model can generate in response
- **Current value**: 512 tokens
- **Effect of changing**:
  - **Higher values (1024, 2048)**: Allows longer, more detailed explanations and reasoning
  - **Lower values (256, 128)**: Forces more concise, brief answers
- **Recommendation**: Keep at 512 for balanced answer length

#### `temperature: 0.1`
- **What it does**: Controls randomness in token selection (0.0 = deterministic, 1.0 = very random)
- **Current value**: 0.1 (very low, nearly deterministic)
- **Effect of changing**:
  - **Lower (0.0-0.05)**: More deterministic, consistent answers (good for accuracy)
  - **Higher (0.3-0.7)**: More creative but potentially less accurate answers
- **Recommendation**: Keep low (0.1-0.2) for answer accuracy in competitive exams

#### `top_p: 0.9`
- **What it does**: Nucleus sampling - considers only top tokens that sum to this probability
- **Current value**: 0.9 (considers 90% of probability mass)
- **Effect of changing**:
  - **Lower (0.7-0.8)**: More focused, conservative token selection
  - **Higher (0.95-1.0)**: Considers more diverse token options
- **Recommendation**: 0.9 is optimal for balanced diversity

#### `repetition_penalty: 1.2`
- **What it does**: Penalizes repeated tokens/phrases (1.0 = no penalty, >1.0 = penalty)
- **Current value**: 1.2 (moderate penalty)
- **Effect of changing**:
  - **Lower (1.0-1.1)**: May produce more repetitive text
  - **Higher (1.3-1.5)**: Strongly discourages repetition, may affect coherence
- **Recommendation**: 1.1-1.3 range works well

#### `do_sample: true`
- **What it does**: Enables sampling-based generation vs greedy decoding
- **Current value**: true
- **Effect of changing**:
  - **false**: Pure greedy decoding (deterministic, ignores temperature/top_p)
  - **true**: Uses sampling with temperature and top_p
- **Recommendation**: Keep true for controlled randomness

---

## 📄 `qgen.yaml` - Question Agent Configuration

**Purpose**: Controls how the Question Agent generates MCQ questions.

### Current Configuration:
```yaml
max_new_tokens: 1024
temperature: 0.7
top_p: 0.9
repetition_penalty: 1.2
do_sample: true
```

### Parameter Details:

#### `max_new_tokens: 1024`
- **What it does**: Maximum tokens for question generation
- **Current value**: 1024 (double the answer agent)
- **Why higher**: Questions need more tokens for:
  - Question text
  - 4 multiple choice options
  - Detailed explanations
- **Effect of changing**:
  - **Lower (512-768)**: May truncate complex questions or explanations
  - **Higher (1536-2048)**: Allows very detailed questions but slower generation

#### `temperature: 0.7`
- **What it does**: Controls creativity in question generation
- **Current value**: 0.7 (moderately creative)
- **Why higher than answer agent**: Questions benefit from creativity and variety
- **Effect of changing**:
  - **Lower (0.3-0.5)**: More predictable, similar question patterns
  - **Higher (0.8-1.0)**: Very creative but may lose coherence
- **Recommendation**: 0.6-0.8 for good variety without losing quality

#### Other Parameters
Same as answer agent but optimized for question generation context.

---

## 🔧 How to Modify Parameters

### Step 1: Edit the YAML files
```bash
# Edit answer agent config
nano agen.yaml

# Edit question agent config  
nano qgen.yaml
```

### Step 2: Run the agents to see effects
```bash
# Test question generation
python -m agents.question_agent --num_questions 5 --output_file test_questions.json --verbose

# Test answer generation
python -m agents.answer_agent --input_file test_questions.json --output_file test_answers.json --verbose
```

---

## 🎯 Recommended Configurations

### For High Accuracy (Competition Mode)
```yaml
# agen.yaml - Answer Agent
max_new_tokens: 512
temperature: 0.05    # Very deterministic
top_p: 0.8          # More focused
repetition_penalty: 1.1
do_sample: true

# qgen.yaml - Question Agent  
max_new_tokens: 1024
temperature: 0.5     # Moderate creativity
top_p: 0.85         # Focused but diverse
repetition_penalty: 1.2
do_sample: true
```

### For Creative Exploration
```yaml
# agen.yaml - Answer Agent
max_new_tokens: 768
temperature: 0.3     # Some creativity
top_p: 0.9
repetition_penalty: 1.2
do_sample: true

# qgen.yaml - Question Agent
max_new_tokens: 1536
temperature: 0.8     # High creativity
top_p: 0.95         # Very diverse
repetition_penalty: 1.3
do_sample: true
```

### For Speed Optimization
```yaml
# Both files
max_new_tokens: 256  # Shorter responses
temperature: 0.1
top_p: 0.8
repetition_penalty: 1.1
do_sample: false     # Fastest generation
```

---

## 🚨 Important Notes

1. **Changes only affect `/agents` scripts**, not `main.py`
2. **Restart required**: Changes take effect on next script run
3. **Balance trade-offs**: Accuracy vs Creativity vs Speed
4. **Test incrementally**: Make small changes and test results
5. **Backup configs**: Save working configurations before experimenting

---

## 🔍 Troubleshooting

### Issue: Generated text is too repetitive
**Solution**: Increase `repetition_penalty` to 1.3-1.4

### Issue: Answers are inconsistent
**Solution**: Lower `temperature` to 0.05-0.1

### Issue: Questions lack variety
**Solution**: Increase `temperature` to 0.8-0.9 in `qgen.yaml`

### Issue: Generation is too slow
**Solution**: 
- Reduce `max_new_tokens`
- Set `do_sample: false`
- Lower `top_p` to 0.7-0.8

### Issue: Responses are cut off
**Solution**: Increase `max_new_tokens` appropriately

---

## 📊 Performance Impact

| Parameter | Higher Value Impact | Lower Value Impact |
|-----------|-------------------|-------------------|
| `max_new_tokens` | Slower, more detailed | Faster, more concise |
| `temperature` | More creative, less consistent | More consistent, less creative |
| `top_p` | More diverse vocabulary | More focused vocabulary |
| `repetition_penalty` | Less repetition, may hurt flow | More repetition possible |
| `do_sample` | Controlled randomness | Pure deterministic |

---

*Last updated: 2025-07-19*
*For technical support, refer to the `/agents` folder README*
