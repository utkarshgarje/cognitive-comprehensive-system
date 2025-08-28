TRANSCRIPT_SUMMARY_TEMPLATE = """Given the following lecture transcript, summarise the content.

# Transcript
{transcript}

# Summary"""

CLEAN_TRANSCRIPT_TEMPLATE = """Given is a raw transcript of a video. It was made by concatenating transcriptions from chunks of the full video text. Clean the trancsript by making the transition between the chunks smooth and coherent. Below is an example of how this might look:

<example>
# Raw transcript
The speaker welcomes everyone and introduces the topic for the lecture. The content of the lecture is not provided in the given transcript. The lecture will explore the history of the Aztec civilization. The lecture discusses the history of the Aztec Empire, starting from its humble beginnings. The lecture discusses the rise and subsequent decline of a subject from its modest origins to its dramatic downfall.

# Cleaned transcript
The speaker welcomes everyone and introduces the topic for the lecture. The lecture will explore the history of the Aztec civilization, starting from its humble beginnings to its dramatic fall.
</example>

# Raw transcript
{transcript}

# Cleaned transcript"""

# --- MODIFIED SUBJECTS_TEMPLATE ---
SUBJECTS_TEMPLATE = """Analyze the following transcript and identify the main subjects or topics discussed.

# Transcript
{transcript}

--- END TRANSCRIPT ---

You must generate a list of the top 5-10 subjects. Follow these critical formatting instructions precisely:

1.  Your entire response MUST be ONLY a single, valid JSON object.
2.  The JSON object MUST have a single key named "subjects".
3.  The value for the "subjects" key MUST be a JSON array of strings.
4.  Each string in the array should be a concise name for a main subject identified in the transcript.
5.  DO NOT include any introductory text, explanations, apologies, commentary, or markdown formatting (like ```json ... ``` or numbering) in your response.
6.  Output ONLY the raw JSON object.

<example>
{{
    "subjects": [
        "The Migration and Foundation of Tenochtitlán",
        "Structure and Hierarchy of Aztec Society",
        "Architectural Achievements of the Aztecs",
        "Aztec Contributions to Astronomy and Calendar Systems",
        "The Rich Cultural Heritage of the Aztecs (Art, Music, Poetry)",
        "Religious Beliefs and Practices in Aztec Society",
        "The Arrival of Hernán Cortés and the Spanish Conquistadors",
        "The Conquest and Fall of the Aztec Empire",
        "The Legacy of the Aztec Empire in Modern Mexican Culture",
        "Overview of Aztec Scientific Achievements"
    ]
}}
</example>

# Subjects JSON Output:
""" # Using a more specific output marker


QUIZ_TEMPLATE = """Given the following lecture transcript. Generate a quiz based on the content. The format of the quiz should be a list of questions, answers and the correct answer. The correct answer should be the index of the correct answer in the list of answers (0-indexed).

Scramble the order of the correct answer so it doesn't always appear first! Don't generate more than 5 items pairs.

See the example format below:
<example>
{{
    "question_and_answers": [
        {{
            "question": "What is the capital of France?",
            "answers": ["London", "Berlin", "Madrid", "Paris"],
            "correct_answer": 3
        }},
        {{
            "question": "What is the capital of Germany?",
            "answers": ["Paris", "London", "Berlin", "Madrid"],
            "correct_answer": 2
        }}
    ]
}}
</example>

# Transcript
{transcript}

# Quiz"""


FLASHCARDS_TEMPLATE = """Given the following lecture transcript. Generate a list of flashcards based on the content. The format of the flashcards should be a list of question, answer and an image prompt that will be sent to Stable Diffusion for image generation. An example of the format is shown below in example tags. Generate up to at most 5 flashcards.

Make the questions specific to the topic and not too general. Don't ask about the professor and the general format of the course and things like that.

<example>
{{
    "flashcards": [
        {{
            "question": "What is the capital of France?",
            "answer": "Paris",
            "image_prompt": "An image of the city Paris in France"
        }},
    ]
}}
</example>

# Transcript
{transcript}

# Flashcards"""