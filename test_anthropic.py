import os
from dotenv import load_dotenv

load_dotenv()

import anthropic
try:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model='claude-opus-4-20250805',
        max_tokens=10,
        messages=[{'role': 'user', 'content': 'test'}]
    )
    print('✓ Anthropic API key works!')
except Exception as e:
    print(f'✗ Anthropic API key error: {e}')
