from gemini_client import get_client

# FIXED: flash-lite instead of flash — 3x cheaper, same quality for classification
MODEL_NAME = "models/gemini-2.5-flash-lite"


def classify_batch(requirements, prompt_template, iso_context):
    client = get_client()  # cached — no repeated initialization

    req_block = "\n".join(requirements)

    prompt = prompt_template.format(
        ISO_CONTEXT=iso_context,
        REQUIREMENTS=req_block,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.0,
            "thinking_config": {"thinking_budget": 0}  # thinking OFF — saves tokens
        }
    )

    return response.text