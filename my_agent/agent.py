import types
import re
import adk
from google.adk.agents.llm_agent import Agent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool, AgentTool
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def parse_and_clean_code(raw_code: str) -> str:
    """
    An ADK-compatible function that takes the raw code string
    from the previous agent, cleans it, and returns the
    clean code string.
    """
    print("Running code parser...")  # For debugging

    if not isinstance(raw_code, str):
        raw_code = str(raw_code)

    clean_text = raw_code.strip()

    # Regex to find code inside markdown backticks
    code_regex = re.compile(r'```(?:[a-zA-Z0-9_]*)\s*([\s\S]*?)\s*```', re.DOTALL)

    match = code_regex.search(clean_text)

    if match:
        # Found code in backticks, return it.
        clean_code = match.group(1).strip()
    else:
        # No backticks found, assume the entire response is the raw code.
        clean_code = clean_text

    # Return the clean string.
    # This will be the final output of the root_agent.
    return clean_code


ui_ux_analyst = Agent(
    model=Gemini(
        model="gemini-2.5-flash-preview-09-2025",
        retry_options=retry_config
    ),
    name='ui_ux_analyst',
    description='A helpful assistant for describing image.',
    instruction='''
            You are a senior UI/UX analyst.
            Analyze this webpage screenshot and provide a detailed, structured description 
            of all UI components, their layout, text content, colors, and fonts.
            This description will be used by an AI code generation agent, so be very specific 
            and describe it like a spec.
        ''',
    output_key="ui_description",
)

CODE_GENERATION_TEMPLATE = """
You are an expert React/TypeScript developer. Your task is to generate a single-file React component (TSX) based on the description of a webpage below.
- Use functional components.
- Use Tailwind CSS for all styling. Do not use inline styles or CSS files.
- Create a self-contained component. I only want the layout, do not try to implement functionality. 
- Do not use hooks like useState
- Do not include React imports (like 'import React') or any imports.
- Use simple SVGs if icons are mentioned, do not import any icons from any package.
- Ensure the component is complete.
- **CRITICAL:** Respond with *only* the raw TSX code for the component which can be directly pasted in react project and should be able to run it. 
- Do not add ```tsx, ```javascript, or any explanations, just the code.
- Start the code with 'const PageComponent = () => {{' and export it at the end.  # <-- THE FIX IS HERE
- Make sure no es lint or ECMAScript issue is present in final output. If present correct it

---
WEBPAGE DESCRIPTION:
{ui_description}
---

Your TSX Code:
"""


def build_code_prompt(ui_description: str) -> str:
    """Combines the UI description with the static code gen rules."""

    final_prompt = CODE_GENERATION_TEMPLATE.format(
        ui_description=ui_description  # <-- FIX: Was 'description='
    )
    print(final_prompt)
    return final_prompt


code_generator = Agent(
    model=Gemini(
        model="gemini-2.5-flash-preview-09-2025",
        retry_options=retry_config,
        generation_config=types.GenerationConfig(
            temperature=0.1,
            stop_sequences=['```']
        )
    ),
    name='code_drafter',
    description="A specialist code generation agent. It executes the detailed prompt it is given to write TSX code.",
    instruction='''
    You are a code generation agent. 
    You will be given a detailed prompt with rules for generating code.
    You must follow those rules precisely.
    Respond with *only* the raw code. Do not add any other text.
    ''',
    output_key="draft_code",
)

root_agent = Agent(
    name='CodeGeneratorPipeline',
    model=Gemini(
        model="gemini-2.5-flash-preview-09-2025",
        retry_options=retry_config
    ),
    description=(
        "A root agent that generates code from a user request (which should include an image). "
        "**CRITICAL INSTRUCTIONS: You must follow this data flow:** "
        "1. **Get Description:** Call the 'ui_ux_analyst' tool with the user's request to get a `ui_description`. "
        "2. **Build Prompt:** Take the `ui_description` and pass it to the 'build_code_prompt' tool to get the final `code_prompt`. "
        "3. **Draft Code:** Pass the `code_prompt` (from the previous step) to the 'code_generator' tool to get the `draft_code`. "
        "4. **Clean Code:** Pass the `draft_code` to the 'parse_and_clean_code' tool to get the final `clean_code`. "
        "5. **Final Output:** Return *only* the `clean_code`."
    ),
    tools=[
        AgentTool(agent=ui_ux_analyst),
        build_code_prompt,
        AgentTool(agent=code_generator),
        parse_and_clean_code
    ],
)