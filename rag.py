import os
import asyncio
import httpx # Use async HTTP client
import time # For adding delays
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.chains import (
    create_chat_chain,
    create_flashcard_chain,
    create_quiz_chain,
    create_subjects_chain,
    create_transcript_summary_chain,
    create_clean_transcript_chain
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from rag.data_models import FlashCards, QuestionAndAnswer, QuestionsAndAnswers # Import data models
from google.api_core.exceptions import ResourceExhausted # Import for potential direct use/checking

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
if not GEMINI_API_KEY:
     raise ValueError("GEMINI_API_KEY environment variable not set.")
if not STABILITY_API_KEY:
     raise ValueError("STABILITY_API_KEY environment variable not set.")

# LLM Initialization (consider adding max_retries if needed, though tenacity handles some)
llm = ChatGoogleGenerativeAI(
     model="gemini-1.5-flash",
     temperature=0,
     request_timeout=120, # Increased timeout for potentially longer generations
     google_api_key=GEMINI_API_KEY,
     # max_retries=1 # Example: reduce internal retries if implementing custom ones
)

# Stability AI Configuration
stability_api_url = "https://api.stability.ai"
stability_model = "stable-diffusion-v1-6" # Or use newer models like "stable-diffusion-xl-1024-v1-0" if available/preferred


# Chain Initialization
try:
    chat_chain = create_chat_chain(llm)
    summary_chain = create_transcript_summary_chain(llm)
    flashcard_chain = create_flashcard_chain(llm)
    quiz_chain = create_quiz_chain(llm)
    subjects_chain = create_subjects_chain(llm)
    response_clean_chain = create_clean_transcript_chain(llm)
except Exception as e:
     print(f"❌ Error initializing Langchain chains: {e}")
     # Depending on severity, you might want to exit or disable features
     raise RuntimeError("Failed to initialize core LLM chains.") from e


# Constants
CHUNK_SIZE = 18000 # Slightly reduced chunk size to be safer with token limits
CHUNK_OVERLAP = 500
# Delay between concurrent API calls in seconds (adjust as needed)
CONCURRENT_API_CALL_DELAY = 1.5 # ~40 RPM if calls take minimal time


def split_transcript(transcript: str):
    """Splits the transcript into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    # Returns List[Document], but we often just need the page_content
    docs = text_splitter.create_documents([transcript])
    return [doc.page_content for doc in docs] # Return list of strings


async def generate_chat_response(input_text: str) -> str:
    """Generates a response using the chat chain."""
    # Add retry logic here if needed beyond what Langchain/Google client provides
    response = await chat_chain.ainvoke({"input": input_text})
    return response


async def split_transcript_and_collect_responses(transcript: str, chain: Runnable) -> list:
    """
    Splits transcript, invokes chain on each chunk CONCURRENTLY with DELAYS,
    and collects responses.
    """
    texts = split_transcript(transcript)
    if not texts:
        return []

    tasks = []
    for i, text_chunk in enumerate(texts):
        # Create the task
        task = asyncio.create_task(chain.ainvoke({"transcript": text_chunk}))
        tasks.append(task)
        # Introduce delay *after* creating the task, before the next iteration
        if i < len(texts) - 1: # Don't delay after the last task
             await asyncio.sleep(CONCURRENT_API_CALL_DELAY)

    # Gather results from all tasks
    # Use return_exceptions=True to catch individual failures
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Process responses, handling potential exceptions
    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            # Log the error and decide how to handle (skip, default value, raise?)
            print(f"⚠️ Error processing chunk {i} for chain {chain.config.get('run_name', 'UnknownChain')}: {response}")
            # Option 1: Skip the chunk
            # continue
            # Option 2: Raise a higher-level exception (might stop the whole process)
            # raise response
            # Option 3: Append a marker or default value (depends on downstream use)
            processed_responses.append(None) # Or some other indicator
        else:
            processed_responses.append(response)

    return processed_responses


async def generate_transcript_summary(transcript: str) -> str:
    """Generates a summary by processing chunks and combining."""
    print(f"Generating summary for transcript of length {len(transcript)}...")
    # Use the concurrent-with-delay function
    responses = await split_transcript_and_collect_responses(transcript, summary_chain)

    # Filter out None values (if skipped chunks) before joining
    valid_summaries = [r for r in responses if r is not None and isinstance(r, str)]

    if not valid_summaries:
         print("⚠️ No valid summary chunks were generated.")
         return "Could not generate summary."

    full_summary = ' '.join(valid_summaries)
    print("Cleaning combined summary...")
    # Cleaning might also hit rate limits if the combined summary is huge
    # Consider if cleaning needs chunking too, or if it's usually short enough
    try:
        output = await response_clean_chain.ainvoke({"transcript": full_summary})
        print("✅ Summary generation complete.")
        return output
    except ResourceExhausted as e:
         print(f"❌ Rate limit hit during summary cleaning: {e}")
         # Return the uncleaned summary as fallback?
         return f"[Uncleaned Summary due to rate limit]: {full_summary}"
    except Exception as e:
        print(f"❌ Error during summary cleaning: {e}")
        return f"[Summary cleaning failed]: {full_summary}"


async def generate_flashcards(transcript: str) -> FlashCards | None:
    """
    Generates flashcards from the transcript.
    NOTE: This currently processes the whole transcript at once.
    Consider chunking if transcripts are very long and cause token/timeout issues,
    but be mindful that chunking increases the *number* of API calls.
    """
    print(f"Requesting flashcards generation for transcript length {len(transcript)}...")
    if flashcard_chain is None:
        raise ValueError("flashcard_chain is not initialized")

    # For flashcards, invoking on the whole transcript might be okay unless it's massive
    # If very long transcripts cause issues (tokens/timeouts), chunking is an option:
    # responses = await split_transcript_and_collect_responses(transcript, flashcard_chain)
    # combined_flashcards = FlashCards(flashcards=[fc for resp in responses if resp for fc in resp.flashcards])
    # return combined_flashcards
    # --- Current single invocation ---
    try:
        # Using ainvoke for consistency, even if not chunking here
        response = await flashcard_chain.ainvoke({"transcript": transcript})
        print("✅ Flashcard generation call complete.")
        if response is None:
             print("⚠️ flashcard_chain.ainvoke returned None")
             return None # Or empty FlashCards(flashcards=[])
        if not isinstance(response, FlashCards):
             print(f"⚠️ Flashcard chain returned unexpected type: {type(response)}")
             # Attempt to handle if it's dict-like, otherwise fail
             if isinstance(response, dict) and 'flashcards' in response:
                  # Manually create the Pydantic object if possible (risky)
                  try:
                       return FlashCards(**response)
                  except Exception:
                       print("❌ Failed to parse dict into FlashCards object.")
                       return None
             return None
        return response
    except Exception as e:
        # Let the specific ResourceExhausted be caught by the endpoint handler
        print(f"❌ Error invoking flashcard chain: {e}")
        raise # Re-raise the exception to be handled by the caller


async def generate_image(prompt: str) -> httpx.Response | None:
    """Generates an image using Stability AI asynchronously."""
    print(f"  -> Calling Stability AI for prompt: '{prompt[:60]}...'")
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [{"text": prompt, "weight": 1.0}], # Weight 1.0 usually better unless combining prompts
        "cfg_scale": 7,
        "height": 512, # Consider 512x512 for better quality if needed
        "width": 512,
        "samples": 1,
        "steps": 30, # Default is often 50, 30 is faster, adjust as needed
    }
    # Increased timeout for image generation
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                f"{stability_api_url}/v1/generation/{stability_model}/text-to-image",
                headers=headers,
                json=payload,
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            print(f"  ✅ Stability AI call successful (Status: {response.status_code}) for prompt: '{prompt[:60]}...'")
            return response
        except httpx.HTTPStatusError as e:
            print(f"❌ Stability AI HTTP Error: {e.response.status_code} - {e.response.text}")
            return e.response # Return response object even on error for inspection
        except httpx.RequestError as e:
            print(f"❌ Stability AI Request Error: {e}")
            return None # Network or connection error
        except Exception as e:
            print(f"❌ Unexpected error in generate_image: {e}")
            return None


async def generate_quiz(transcript: str) -> list[dict]:
    """
    Generates quiz questions by processing chunks concurrently with delays
    and combining the results.
    """
    print(f"Generating quiz for transcript of length {len(transcript)}...")
    # Use the concurrent-with-delay function
    responses = await split_transcript_and_collect_responses(transcript, quiz_chain)

    if responses is None: # Should not happen with return_exceptions=True, but check anyway
        print("⚠️ split_transcript_and_collect_responses returned None unexpectedly.")
        raise ValueError("Quiz generation failed during chunk processing.")

    flattened_list = []
    for i, response in enumerate(responses):
        # Skip chunks that failed or returned None
        if response is None or isinstance(response, Exception):
            print(f"ℹ️ Skipping quiz results from failed chunk {i}.")
            continue

        # Validate the structure of each successful response
        if not isinstance(response, QuestionsAndAnswers):
             print(f"⚠️ Quiz chain (chunk {i}) returned unexpected type: {type(response)}. Expected QuestionsAndAnswers.")
             continue
        if not hasattr(response, "question_and_answers") or not isinstance(response.question_and_answers, list):
            print(f"⚠️ Malformed quiz response from chunk {i}: missing or invalid 'question_and_answers' attribute.")
            continue

        for item in response.question_and_answers:
             # Ensure each item is the expected Pydantic model or can be converted
             if isinstance(item, QuestionAndAnswer):
                  # Convert Pydantic model to dict for JSON serialization by FastAPI
                  flattened_list.append(item.dict()) # Use .model_dump() in Pydantic v2
             elif isinstance(item, dict):
                  # Basic check if it looks like the expected structure
                  if 'question' in item and 'answers' in item and 'correct_answer' in item:
                       flattened_list.append(item)
                  else:
                       print(f"⚠️ Skipping malformed quiz item (dict) from chunk {i}: {item}")
             else:
                   print(f"⚠️ Skipping unexpected quiz item type ({type(item)}) from chunk {i}: {item}")

    print(f"✅ Quiz generation complete. Generated {len(flattened_list)} questions.")
    return flattened_list


async def generate_subjects(transcript: str) -> list[str]:
    """
    Generates subjects by processing chunks concurrently with delays
    and combining the results.
    """
    print(f"Generating subjects for transcript of length {len(transcript)}...")
    # Use the concurrent-with-delay function
    responses = await split_transcript_and_collect_responses(transcript, subjects_chain)

    if responses is None:
         print("⚠️ split_transcript_and_collect_responses returned None unexpectedly for subjects.")
         raise ValueError("Subject generation failed during chunk processing.")

    flattened_list = []
    for i, response in enumerate(responses):
         # Skip chunks that failed or returned None
         if response is None or isinstance(response, Exception):
             print(f"ℹ️ Skipping subjects results from failed chunk {i}.")
             continue

         # Expecting a dictionary with a "subjects" key containing a list of strings
         if isinstance(response, dict) and "subjects" in response and isinstance(response["subjects"], list):
             subjects_in_chunk = [str(item) for item in response["subjects"]] # Ensure strings
             flattened_list.extend(subjects_in_chunk)
         else:
             print(f"⚠️ Malformed subjects response from chunk {i}: {response}")

    # Remove duplicates while preserving order (optional)
    unique_subjects = list(dict.fromkeys(flattened_list))

    print(f"✅ Subject generation complete. Found {len(unique_subjects)} unique subjects.")
    return unique_subjects



# import os

# import requests
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# from rag.chains import (create_chat_chain, create_flashcard_chain,
#                         create_quiz_chain, create_subjects_chain,
#                         create_transcript_summary_chain,
#                         create_clean_transcript_chain)
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import asyncio
# import os
# from langchain_core.runnables import Runnable
# from rag.data_models import QuestionAndAnswer

# load_dotenv()

# llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, request_timeout=20, max_retries=10)
# stability_api = "https://api.stability.ai"
# stability_model = "stable-diffusion-v1-6"

# chat_chain = create_chat_chain(llm)
# summary_chain = create_transcript_summary_chain(llm)
# flashcard_chain = create_flashcard_chain(llm)
# quiz_chain = create_quiz_chain(llm)
# subjects_chain = create_subjects_chain(llm)
# response_clean_chain = create_clean_transcript_chain(llm)

# CHUNK_SIZE = 20_000

# def split_transcript(transcript: str):
#     text_splitter = RecursiveCharacterTextSplitter(
#         # Set a really small chunk size, just to show.
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=200,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     texts = text_splitter.create_documents([transcript])
#     return texts


# def generate_chat_response(input_text: str):
#     response = chat_chain.invoke({"input": input_text})
#     return response


# async def split_transcript_and_collect_responses(transcript: str, chain: Runnable):
#     texts = split_transcript(transcript)
#     tasks = [asyncio.create_task(chain.ainvoke({"transcript": text})) for text in texts]
#     responses = await asyncio.gather(*tasks)
#     return responses


# async def generate_transcript_summary(transcript: str):
#     responses = await split_transcript_and_collect_responses(transcript, summary_chain)
#     full_summary = ' '.join(responses)

#     output = response_clean_chain.invoke({"transcript": full_summary})
#     return output


# async def generate_flashcards(transcript: str):
#     response = flashcard_chain.invoke({"transcript": transcript})
#     return response


# async def generate_image(prompt, stability_api=stability_api, stability_model=stability_model):
#     response = requests.post(
#         f"{stability_api}/v1/generation/{stability_model}/text-to-image",
#         headers={
#             "Content-Type": "application/json",
#             "Accept": "application/json",
#             "Authorization": f"Bearer {os.environ.get('STABILITY_API_KEY')}"
#         },
#         json={
#             "text_prompts": [
#                 {
#                     "text": prompt,
#                     "weight" : 0.5
#                 }
#             ],
#             "cfg_scale": 7,
#             "height": 320,
#             "width": 320,
#             "samples": 1,
#         },
#     )
#     return response


# async def generate_quiz(transcript: str):
#     responses = await split_transcript_and_collect_responses(transcript, quiz_chain)
#     sublists = [response.question_and_answers for response in responses]
#     flattened_list = [QuestionAndAnswer(question=item.question, answers=item.answers, correct_answer=item.correct_answer).json() for sublist in sublists for item in sublist]
#     return flattened_list


# async def generate_subjects(transcript: str):
#     responses = await split_transcript_and_collect_responses(transcript, subjects_chain)
#     subject_lists = [response["subjects"] for response in responses]
#     flattened_list = [item for sublist in subject_lists for item in sublist]
#     return flattened_list
