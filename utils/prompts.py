from langchain import PromptTemplate, LLMChain

prompt_template = """Your job is to write a summary of a text.

Here are the rules:
All relevant information, anecdotes, facts, etc must appear in your summary.
You have to use markdown bulletpoints to format your summary.
The bulletpoints can be indented to help organize information hierarchically.
Your summary will be written in the same language as the input text.
Irrelevant information are to be discarded, for example sponsors, advertissement, embellishment, headers etc

Here's the first part of the text:
'''
{text}
'''

EXECUTIVE SUMMARY IN MARKDOWN AND WITHOUT TRANSLATION:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    """Your job is to continue the summary of a text.

Here are the rules:
All relevant information, anecdotes, facts, etc must appear in your summary.
You have to use markdown bulletpoints to format your summary.
The bulletpoints can be indented to help organize information hierarchically.
Your summary will be written in the same language as the input text.
Irrelevant information are to be discarded, for example sponsors, advertissement, embellishment, headers etc

Here's the summary so far:
'''
{existing_answer}
'''

Here's the next section of the text:
'''
{text}
'''

Given this new section of the text, refine the summary. If no changes are needed, simply answer the original summary.

EXECUTIVE SUMMARY IN MARKDOWN AND WITHOUT TRANSLATION:
"""
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)
