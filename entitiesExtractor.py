from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import json
import re

load_dotenv()

def get_few_shot_examples():
    return json.load(open("example_10_shot.json"))


def generate_prompt():

    entity_definition = [
        {
            "entity": "skill",
            "definitions": "Any mention of a skill or ability or tool or proficiency that is relevant to a job or task.",
        },
        {
            "entity": "responsibility",
            "definitions": "Any mention of a responsibility or duty or task that have been done that is relevant to a job or task.",
        },
        {
            "entity": "degree",
            "definitions": "Any mention of a degree or qualification or certification that is relevant to a job or task.",
        },
        {
            "entity": "experience",
            "definitions": "Any mention of job experience or job position with or without the job title that is relevant to a job or task.",
        }
    ]

    labels = [item['entity'] for item in entity_definition]
    label_definitions = {item['entity']: item['definitions'] for item in entity_definition}

    definitions_str = '\n'.join([f"{label}: {definition}" for label, definition in label_definitions.items()])

    template = f"""
    You are an expert Named Entity Recognition (NER) system.
    Your task is to accept Text as input and extract named entities.
    Entities could be a word or a span must have one of the following labels: {', '.join(labels)}.

    Below are definitions of each label to help aid you in what kinds of named entities to extract for each label.
    Assume these definitions are written by an expert and follow them closely.

    {definitions_str}

    The output should be a JSON object where each key is a label from one of the following labels: {', '.join(labels)}, he value associated with each label is a list of objects.
    Each object within the list must only has one key : entity (A string representing the extracted entity from the input sentence). Do not include any other keys in the object.
    The value of the entity key should not contain any single or double quotes.
    Only use this output format but use the labels provided above instead of the ones defined in the example below.

    Important: Do not output anything besides the JSON object. Do not include any introductory text, conclusions, or additional commentary.
    """
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=get_few_shot_examples(),
        example_prompt=example_prompt,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    return final_prompt

def convert_quotes(s):
    # Replace single quotes after '{' or before '}'
    s = re.sub(r"(?<=\{)'|'(?=\})", '"', s)
    
    # Replace single quotes before ':'
    s = re.sub(r"'(?=:)", '"', s)
    
    # Replace single quotes after a whitespace
    s = re.sub(r"(?<=\s)'", '"', s)

    # Replace single quotes before ','
    s = re.sub(r"'(?=,)", '"', s)

    s = re.sub(r"(?<=[^\s])[/\\](?=[^\s])", "", s)

    return s

def parse_entities(entities):

    entities_list = eval(entities)

    labels = ["skill", "responsibility", "degree", "experience"]

    categorized_entities = {label: [] for label in labels}

    for label in labels:
        upper_label = label.upper()
        if upper_label in entities_list:
            entities = [item["entity"] for item in entities_list[upper_label]]
            categorized_entities[label].extend(entities)
        if label in entities_list:
            entities = [item["entity"] for item in entities_list[label]]
            categorized_entities[label].extend(entities)

    return categorized_entities

def parse_entities_v2(entities):

    start_index = entities.find("{")

    end_index = entities.rfind("}")

    clean_entities = entities[start_index:end_index+1]

    converted_entities = convert_quotes(clean_entities)

    converted_json = json.loads(converted_entities)

    labels = ["skill", "responsibility", "degree", "experience"]

    categorized_entities = {label: [] for label in labels}

    for label in labels:
        upper_label = label.upper()
        if upper_label in converted_json:
            entities = [item["entity"] for item in converted_json[upper_label]]
            categorized_entities[label].extend(entities)
        if label in converted_json:
            entities = [item["entity"] for item in converted_json[label]]
            categorized_entities[label].extend(entities)

    return categorized_entities

def mistral_parse_entities(entities):
    pattern = re.compile(r'^\d+\.\s*(.*?)\s*\|\s*(.*?)\s*\|', re.MULTILINE)

    matches = pattern.findall(entities)

    categorized_entities = {
        "skill": [],
        "responsibility": [],
        "degree": [],
        "experience": []
    }

    for match in matches:
        text = match[0].strip()
        label = match[1].strip().lower()

        if label == "degree":
            categorized_entities["degree"].append(text)
        elif label == "skill":
            categorized_entities["skill"].append(text)
        elif label == "responsibility":
            categorized_entities["responsibility"].append(text)
        elif label == "experience":
            categorized_entities["experience"].append(text)

    return categorized_entities

def Gpt_infer(input):
    final_prompt = generate_prompt()
    chain = final_prompt | ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
    for _ in range(2):  # Attempt to retry once if an error occurs
        try:
            output = chain.invoke({"input": input})
            parsed_entities = parse_entities_v2(output.content)
            total_token = output.usage_metadata["total_tokens"]
            return {
                "entities": parsed_entities,
                "total_token": total_token
            }
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
    
    # If it fails after retries, raise the exception
    raise Exception("Failed to parse entities after retries.")

def Claude_infer(input):
    final_prompt = generate_prompt()
    # chain = final_prompt | ChatAnthropic(model="claude-3-opus-20240229", temperature=0.2, max_tokens=4096, timeout=None, max_retries=2)
    chain = final_prompt | ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=4096, timeout=None, max_retries=2)
    # chain = final_prompt | ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2, max_tokens=4096, timeout=None, max_retries=2)

    for _ in range(2):  # Attempt to retry once if an error occurs
        try:
            output = chain.invoke({"input": input})
            parsed_entities = parse_entities_v2(output.content)
            total_token = output.usage_metadata["total_tokens"]
            return {
                "entities": parsed_entities,
                "total_token": total_token
            }
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
    
    # If it fails after retries, raise the exception
    raise Exception("Failed to parse entities after retries.")

def Mistral_infer(input):
    final_prompt = generate_prompt()
    chain = final_prompt | ChatMistralAI(model="open-mistral-7b", temperature=0.2, max_tokens=4096, timeout=120, max_retries=2)
    # chain = final_prompt | ChatMistralAI(model="open-mixtral-8x22b", temperature=0.2, max_tokens=4096, timeout=120, max_retries=2)

    for _ in range(2):  # Attempt to retry once if an error occurs
        try:
            output = chain.invoke({"input": input})
            parsed_entities = parse_entities_v2(output.content)
            total_token = output.response_metadata["token_usage"]["total_tokens"]
            return {
                "entities": parsed_entities,
                "total_token": total_token
            }
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
    
    # If it fails after retries, raise the exception
    raise Exception("Failed to parse entities after retries.")

# def moke_prompt():

#     prompt = "From the input text, please extract the following entities: skills, responsibilities, degrees, and experiences"

#     final_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", prompt),
#             ("human", "{input}"),
#         ]
#     )

#     return final_prompt

# def coba_prompt_biasa(input):
#     final_prompt = moke_prompt()
#     chain = final_prompt | ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
#     for _ in range(2):  # Attempt to retry once if an error occurs
#         try:
#             output = chain.invoke({"input": input})
#             total_token = output.usage_metadata["total_tokens"]
#             print(output.content)
#             return {
#                 "entities": output,
#                 "total_token": total_token
#             }
#         except Exception as e:
#             print(f"Error occurred: {e}. Retrying...")
    
#     # If it fails after retries, raise the exception
#     raise Exception("Failed to parse entities after retries.")