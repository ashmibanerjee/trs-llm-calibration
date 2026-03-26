"""Response handler for judge LLM evaluations."""

import json
import re
import time
from typing import Dict

from src.llm_setup.base_model import BaseLLMModel
from src.direct_reasoner.token_counter import get_token_stats


def attempt_json_repair(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # Fix single quotes to double quotes (be careful with apostrophes in content)
    # Only replace single quotes that appear to be JSON delimiters
    json_str = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', json_str)  # Keys
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Values after colons

    # Remove any control characters that might cause issues
    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')

    return json_str.strip()


def clean_json_response(response: str) -> str:
    """
    Clean JSON response by removing markdown and preamble text.

    Args:
        response: Raw response string

    Returns:
        Cleaned JSON string
    """
    response_clean = response.strip()

    # Remove markdown code blocks
    if "```json" in response_clean:
        # Extract content between ```json and ```
        start = response_clean.find("```json") + 7
        end = response_clean.find("```", start)
        if end != -1:
            response_clean = response_clean[start:end].strip()
    elif "```" in response_clean:
        # Extract content between ``` and ```
        start = response_clean.find("```") + 3
        end = response_clean.find("```", start)
        if end != -1:
            response_clean = response_clean[start:end].strip()

    # Find the first opening brace/bracket
    first_brace = response_clean.find('{')
    first_bracket = response_clean.find('[')

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        response_clean = response_clean[first_brace:]
    elif first_bracket != -1:
        response_clean = response_clean[first_bracket:]

    # Find the matching closing brace/bracket by counting nesting
    if response_clean.startswith('{'):
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response_clean):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        response_clean = response_clean[:i + 1]
                        break

    elif response_clean.startswith('['):
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response_clean):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        response_clean = response_clean[:i + 1]
                        break

    return response_clean.strip()


def evaluate_recommendations(
    judge_model: BaseLLMModel,
    system_prompt: str,
    user_prompt: str,
    query: str,
    query_id: str,
    model_name: str,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Dict:
    """
    Evaluate recommendations using the judge LLM.

    Args:
        judge_model: Judge model instance
        system_prompt: System instruction for judge
        user_prompt: Formatted user prompt
        query: Original query
        query_id: Query identifier
        model_name: Judge model name for token counting
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries

    Returns:
        Evaluation result dictionary
    """
    response_text = None
    full_input = f"{system_prompt}\n\n{user_prompt}"
    enhanced_system = system_prompt

    for attempt in range(max_retries):
        try:
            # Enhance prompt on retry
            if attempt > 0:
                enhanced_system = system_prompt + "\n\nIMPORTANT: Your previous response had JSON formatting issues. Please ensure:\n1. Return ONLY valid JSON\n2. Use double quotes for all strings and keys\n3. No trailing commas\n4. Proper escaping of special characters"
                full_input = f"{enhanced_system}\n\n{user_prompt}"
                print(f"    → Retry attempt {attempt + 1} with enhanced instructions")

            # Generate evaluation
            response_text = judge_model.generate(
                enhanced_system,
                user_prompt
            )

            # Calculate token stats
            token_stats = get_token_stats(full_input, response_text, model_name)

            # Parse JSON
            response_clean = clean_json_response(response_text)
            evaluation = json.loads(response_clean)

            # Success
            return {
                'query_id': query_id,
                'query': query,
                'evaluation': evaluation,
                'raw_response': response_text,
                'token_stats': token_stats,
                'success': True
            }

        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON decode error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")

            # Try to repair the JSON
            if response_text and attempt < max_retries - 1:
                print(f"    → Attempting JSON repair...")
                try:
                    response_repaired = attempt_json_repair(clean_json_response(response_text))
                    evaluation = json.loads(response_repaired)

                    # Repair successful
                    token_stats = get_token_stats(full_input, response_text, model_name)
                    print(f"    ✓ JSON repair successful!")
                    return {
                        'query_id': query_id,
                        'query': query,
                        'evaluation': evaluation,
                        'raw_response': response_text,
                        'token_stats': token_stats,
                        'success': True,
                        'repaired': True
                    }
                except json.JSONDecodeError:
                    print(f"    ✗ JSON repair failed")
                    # Continue to retry logic below
                    pass

            if attempt < max_retries - 1:
                print(f"    → Sleeping {retry_delay}s before retry...")
                time.sleep(retry_delay)
                continue
            else:
                # Final attempt failed - save problematic response for debugging
                token_stats = get_token_stats(full_input, response_text or "", model_name)

                # Try one last time with aggressive repair
                if response_text:
                    try:
                        print(f"    → Final JSON repair attempt...")
                        response_repaired = attempt_json_repair(clean_json_response(response_text))
                        evaluation = json.loads(response_repaired)
                        print(f"    ✓ Final repair successful!")
                        return {
                            'query_id': query_id,
                            'query': query,
                            'evaluation': evaluation,
                            'raw_response': response_text,
                            'token_stats': token_stats,
                            'success': True,
                            'repaired': True
                        }
                    except:
                        pass

                print(f"    ✗ All repair attempts failed")
                return {
                    'query_id': query_id,
                    'query': query,
                    'evaluation': None,
                    'raw_response': response_text or '',
                    'error': f'JSON decode error: {str(e)}',
                    'token_stats': token_stats,
                    'success': False
                }

        except Exception as e:
            print(f"    ✗ Error: {str(e)[:100]}")
            token_stats = get_token_stats(full_input, response_text or "", model_name)
            return {
                'query_id': query_id,
                'query': query,
                'evaluation': None,
                'raw_response': response_text or '',
                'error': str(e),
                'token_stats': token_stats,
                'success': False
            }

    return {
        'query_id': query_id,
        'query': query,
        'evaluation': None,
        'error': 'Max retries exceeded',
        'token_stats': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
        'success': False
    }
