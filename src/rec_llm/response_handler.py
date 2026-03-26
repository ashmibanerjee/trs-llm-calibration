"""Response processing utilities."""

import json
import time
from typing import Dict

from src.llm_setup.base_model import BaseLLMModel


def clean_json_response(response: str) -> str:
    """
    Clean JSON response by removing markdown code blocks and any preamble text.

    Args:
        response: Raw response string

    Returns:
        Cleaned response string
    """
    response_clean = response.strip()

    # Remove any text before the first { or [
    first_brace = response_clean.find('{')
    first_bracket = response_clean.find('[')

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        # JSON object starts first
        response_clean = response_clean[first_brace:]
    elif first_bracket != -1:
        # JSON array starts first
        response_clean = response_clean[first_bracket:]

    # Remove markdown code blocks if they exist
    if response_clean.startswith("```json"):
        response_clean = response_clean[7:]
    elif response_clean.startswith("```"):
        response_clean = response_clean[3:]

    if response_clean.endswith("```"):
        response_clean = response_clean[:-3]

    # Remove any trailing text after the last } or ]
    last_brace = response_clean.rfind('}')
    last_bracket = response_clean.rfind(']')

    if last_brace != -1 and last_brace > last_bracket:
        response_clean = response_clean[:last_brace + 1]
    elif last_bracket != -1:
        response_clean = response_clean[:last_bracket + 1]

    return response_clean.strip()


def generate_recommendation(
    model: BaseLLMModel,
    system_prompt: str,
    user_prompt: str,
    query: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict:
    """
    Generate a single recommendation with retry logic.

    Args:
        model: BaseLLMModel instance (any provider: Gemini, GPT, Claude, etc.)
        system_prompt: System instruction
        user_prompt: Formatted user prompt
        query: Original query string
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Result dictionary with success status and data/error
    """
    response_text = None

    for attempt in range(max_retries):
        try:
            # On retry attempts after JSON errors, enhance the system prompt
            if attempt > 0:
                enhanced_system_prompt = system_prompt + "\n\nIMPORTANT: Your previous response had JSON formatting issues. Please ensure:\n1. All strings are properly quoted with double quotes\n2. All JSON keys use double quotes, not single quotes\n3. No trailing commas after the last item in arrays or objects\n4. Escape special characters in strings (quotes, backslashes, etc.)\n5. Return ONLY valid, parseable JSON with no extra text"
                print(f"  → Retry attempt {attempt + 1} with enhanced JSON formatting instructions")
            else:
                enhanced_system_prompt = system_prompt

            # Generate response using the model's generate method
            response_text = model.generate(enhanced_system_prompt, user_prompt)

            # BREAKPOINT: Print raw response for debugging
            print("\n" + "="*80)
            print("DEBUG: Raw Response from Model")
            print("="*80)
            print(f"Response type: {type(response_text)}")
            print(f"Response length: {len(response_text) if response_text else 0}")
            print(f"First 500 chars: {response_text[:500] if response_text else 'EMPTY'}")
            print(f"Last 200 chars: {response_text[-200:] if response_text and len(response_text) > 200 else ''}")
            print("="*80 + "\n")

            # Parse JSON
            response_clean = clean_json_response(response_text)

            # BREAKPOINT: Print cleaned response
            print("\n" + "="*80)
            print("DEBUG: Cleaned Response")
            print("="*80)
            print(f"Cleaned length: {len(response_clean) if response_clean else 0}")
            print(f"First 500 chars: {response_clean[:500] if response_clean else 'EMPTY'}")
            print("="*80 + "\n")

            result = json.loads(response_clean)

            # Add metadata
            result['raw_response'] = response_text
            result['success'] = True
            result['query'] = query

            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠ Attempt {attempt + 1}/{max_retries} - JSON decode error: {str(e)[:100]}")
            print(f"  → Error position: line {e.lineno}, column {e.colno}")
            print(f"  → Problematic text around error: {e.doc[max(0, e.pos-50):e.pos+50] if e.doc else 'N/A'}")

            # If not the last attempt, sleep and retry with model regeneration
            if attempt < max_retries - 1:
                print(f"  → Sleeping {retry_delay}s before regenerating with stricter JSON requirements...")
                time.sleep(retry_delay)
                continue
            else:
                # Last attempt failed
                return {
                    'query': query,
                    'raw_response': response_text or '',
                    'error': f'JSON decode error after {max_retries} attempts: {str(e)}',
                    'success': False
                }

        except Exception as e:
            print(f"  ✗ Error generating response: {str(e)[:100]}")
            return {
                'query': query,
                'raw_response': response_text or '',
                'error': str(e),
                'success': False
            }

    return {
        'query': query,
        'error': 'Max retries exceeded',
        'success': False
    }

