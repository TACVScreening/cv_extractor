from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import json

load_dotenv()

def get_few_shot_examples():
    return json.load(open("example.json"))


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

    Your answer must follow the output format below.

    Here is an example of the output format for a paragraph using different labels than this task requires.
    The output format should be as follows: `text | label | explanation`.
    Only use this output format but use the labels provided above instead of the ones defined in the example below.

    Important: Do not output anything besides entities in the specified output format. Do not include any introductory text, conclusions, or additional commentary.
    Output entities in the order they occur in the input paragraph regardless of label.

    Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

    Paragraph: Sriracha sauce goes really well with hoisin stir fry, but you should add it after you use the wok.
    Answer:
    1. Sriracha sauce | INGREDIENT | is an ingredient to add to a stir fry
    2. hoisin stir fry | DISH | is a dish with stir fry vegetables and hoisin sauce
    3. wok | EQUIPMENT | is a piece of cooking equipment used to stir fry ingredients
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

def parse_entities(entities):

    entities_list = eval(entities)

    categorized_entities = {
        "skill": [],
        "responsibility": [],
        "degree": [],
        "experience": []
    }

    for entity in entities_list:
        label = entity['label'].lower()  # Convert label to lowercase to match the keys in categorized_entities
        if label in categorized_entities:
            categorized_entities[label].append(entity['text'])

    return categorized_entities

def infer(input):
    final_prompt = generate_prompt()
    chain = final_prompt | ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
    output =  chain.invoke({"input": input})
    return parse_entities(output.content)

# with open('parsed_entities.json', 'w') as json_file:
#     json.dump(entity, json_file, indent=4)
# prompt = generate_prompt(entity_defintion)
# print(prompt)