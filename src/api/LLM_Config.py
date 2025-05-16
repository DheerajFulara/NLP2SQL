import os
import logging
import re
from dotenv import load_dotenv

# Import both providers
import google.generativeai as genai
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get LLM provider from environment variables with GEMINI as default
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "GEMINI").upper()
logger.info(f"LLM Provider: {LLM_PROVIDER}")

# Validate required environment variables based on the provider
if LLM_PROVIDER == "GEMINI":
    REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]
elif LLM_PROVIDER == "AZURE":
    REQUIRED_ENV_VARS = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
else:
    logger.error(f"Unsupported LLM provider: {LLM_PROVIDER}. Supported providers: GEMINI, AZURE")
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}. Supported providers: GEMINI, AZURE")

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Configure selected provider API
if LLM_PROVIDER == "GEMINI":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
elif LLM_PROVIDER == "AZURE":
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"  # Update as needed

def get_completion_from_gemini(
    system_message: str,
    user_message: str,
    temperature: float = 0.0
) -> str:
    """
    Generate a completion response from Gemini based on the provided system and user messages.

    Returns:
        str: The generated response content.
    """
    try:
        # Combine system and user messages
        combined_message = f"{system_message}\n\nUser Query: {user_message}"
        logger.info("=== INPUT ===")
        logger.info(f"Combined Message:\n{combined_message}")
        logger.info(f"Temperature: {temperature}")
        
        model_instance = genai.GenerativeModel('gemini-2.0-flash')
        response = model_instance.generate_content(
            contents=combined_message,
            generation_config={"temperature": temperature}
        )

        logger.info("=== RAW OUTPUT ===")
        logger.info(f"Response Object: {response}")

        # Ensure text is a string and perform basic cleaning
        text = response.text if isinstance(response.text, str) else str(response.text)
        clean_text = re.sub(r'```json\n|\n```', '', text)

        logger.info("=== CLEANED OUTPUT ===")
        logger.info(f"Cleaned Text:\n{clean_text}")

        return clean_text
    except Exception as e:
        logger.exception(f"Error generating response from Gemini: {str(e)}")
        raise e

def get_completion_from_azure(
    system_message: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> str:
    """
    Generate a completion response from Azure OpenAI based on the provided system and user messages.

    Returns:
        str: The generated response content.
    """
    try:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        logger.info("=== INPUT ===")
        logger.info(f"System Message:\n{system_message}")
        logger.info(f"User Message:\n{user_message}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Deployment: {deployment_name}")

        response = openai.ChatCompletion.create(
            deployment_id=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        logger.info("=== RAW OUTPUT ===")
        logger.info(f"Response Object: {response}")

        text = response["choices"][0]["message"]["content"]
        clean_text = re.sub(r'```json\n|\n```', '', text)

        logger.info("=== CLEANED OUTPUT ===")
        logger.info(f"Cleaned Text:\n{clean_text}")

        return clean_text
    except Exception as e:
        logger.exception(f"Error generating response from Azure OpenAI: {str(e)}")
        raise e

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    n: int = 1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> str:
    """
    Generate a completion response based on the configured provider.
    Routes to the appropriate provider-specific implementation.

    Returns:
        str: The generated response content.
    """
    logger.info(f"Using {LLM_PROVIDER} provider")

    if LLM_PROVIDER == "GEMINI":
        return get_completion_from_gemini(
            system_message=system_message,
            user_message=user_message,
            temperature=temperature
        )
    elif LLM_PROVIDER == "AZURE":
        return get_completion_from_azure(
            system_message=system_message,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    else:
        # This should never happen due to initial validation
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")