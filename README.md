# LLM Input Sanitizer

A Python package for sanitizing and securing user inputs before sending them to large language models (LLMs).

## Features

- **PII Detection & Masking**: Automatically detects and masks emails, phone numbers, SSNs, and credit card numbers
- **Profanity Filtering**: Removes or masks profanity and inappropriate language
- **Input Truncation**: Prevents excessively long inputs
- **Unicode Normalization**: Handles special characters and ensures consistent text encoding
- **Prompt Injection Defense**: Detects and blocks common LLM prompt injection attacks
- **Jailbreak Prevention**: Identifies attempts to bypass LLM safety measures

## Installation

```bash
pip install llm-input-sanitizer
```

## Quick Start

```python
from llm_input_sanitizer import InputSanitizer, prepare_llm_messages, is_input_appropriate

# Initialize the sanitizer
sanitizer = InputSanitizer(max_length=1000)

# Sanitize user input
user_input = "My email is john@example.com and my phone is 555-123-4567"
sanitized_input = sanitizer.sanitize_input(user_input)
# Result: "My email is [EMAIL] and my phone is [PHONE]"

# Check if input is appropriate (no injection attempts)
if is_input_appropriate(sanitized_input):
    # Prepare messages for the LLM
    messages = prepare_llm_messages(
        sanitized_input, 
        system_message="You are a helpful assistant."
    )
    # Send messages to your LLM
else:
    print("Potentially harmful input detected")
```

## Custom Profanity list

```python
sanitizer = InputSanitizer(profanity_file="path/to/profanity_words.txt")
```

## Custom forbidden paths

```python
from llm_input_sanitizer import is_input_appropriate

my_patterns = [
    r'custom_pattern_1',
    r'custom_pattern_2',
]

is_safe = is_input_appropriate(text, forbidden_patterns=my_patterns)
```

## Integration with OpenAI

```python
import openai
from llm_input_sanitizer import InputSanitizer, prepare_llm_messages, is_input_appropriate

sanitizer = InputSanitizer()

def safe_llm_call(user_input):
    # Sanitize the input
    clean_input = sanitizer.sanitize_input(user_input)
    
    # Check if appropriate
    if not is_input_appropriate(clean_input):
        return "I'm sorry, I can't process that request."
    
    # Prepare messages
    messages = prepare_llm_messages(clean_input)
    
    # Call the LLM API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    return response.choices[0].message.content
```

This package provides a baseline of protection against common attacks but is not a complete security solution. Always implement defense in depth for production systems:

- Server-side validation
- Rate limiting
- Monitoring for unusual patterns
- Regular updates to security patterns
