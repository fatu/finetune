import re

def extract_code_from_response(response: str) -> str:
    """
    Extract code from between <code> and </code> tags.
    """
    # Find code between <code> and </code> tags
    code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Fallback: try to find code after the last <think> tag
    think_split = response.split('</think>')
    if len(think_split) > 1:
        potential_code = think_split[-1].strip()
        # Clean up any remaining tags
        potential_code = potential_code.replace('<code>', '').replace('</code>', '').strip()
        if potential_code:
            return potential_code
    
    # Last fallback: return the whole response stripped of tags
    cleaned = response.replace('<think>', '').replace('</think>', '')
    cleaned = cleaned.replace('<code>', '').replace('</code>', '').strip()
    return cleaned

def format_action_history(history: list[tuple[str, str]], max_entries: int = 3) -> str:
    """
    Format the action history for the prompt.
    """
    if not history:
        return "No previous actions."
    
    
    formatted = []
    for i, (code, output) in enumerate(history, 1):
        formatted.append(f"\nStep {i}:")
        formatted.append(f"Code: {code}")
        formatted.append(f"Output: {output[:100000]}..." if len(output) > 100000 else f"Output: {output}")
    
    return "\n".join(formatted)

