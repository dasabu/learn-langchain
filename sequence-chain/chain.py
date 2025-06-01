import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Note: 
# - `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
# - `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-4o-mini',
    temperature=1.0
)

# Story chain
story_template = '''As a story writer, write an English short story (2-3 sentences) based on the given topic and genre.

Topic: {topic}

Genre: {genre}

Story:
'''

story_prompt = PromptTemplate(
    input_variables=['topic', 'genre'],
    template=story_template,
)

story_chain = LLMChain(
    llm=llm,
    prompt=story_prompt,
    output_key='story'
)

# Translate chain
translate_template = '''As a translator, translate the following story into {language}.

Story: {story}

Language: {language}

Translation:
'''

translate_prompt = PromptTemplate(
    input_variables=['story', 'language'],
    template=translate_template,
)

translate_chain = LLMChain(
    llm=llm,
    prompt=translate_prompt,
    output_key='translation'
)

# Sequential chain
chain = SequentialChain(
    chains=[story_chain, translate_chain],
    input_variables=['topic', 'genre', 'language'],
    output_variables=['story', 'translation']
)

# Run chain
def generate_lullaby(topic, genre, language):
    return chain({
        'topic': topic,
        'genre': genre,
        'language': language,
    })
    