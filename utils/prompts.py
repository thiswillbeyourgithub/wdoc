from langchain import PromptTemplate, LLMChain

prompt_template = """Your job is to write a summary of the beginning of a text.
You have to format each information as markdown bullet points.
You have to use indentation to organize those bullet points hierarchically.
You have to write in the same language as the input text.
You have to ignore information not relevant to the topic of the text (sponsors, advertissement, headers, urls, etc)

'''
{text}
'''

SUMMARY OF INFORMATIONS IN MARKDOWN AND IN THE SAME LANGUAGE:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    """You received the summary of the beginning of a text and your job is to continue the summary based on the next part of the text.
You have to format each information as markdown bullet points.
You have to use indentation to organize those bullet points hierarchically.
You have to write in the same language as the input text.
You have to ignore information not relevant to the topic of the text (sponsors, advertissement, headers, urls, etc)

Here's the summary of the text so far:
'''
{existing_answer}
'''

Here's the next part of the text:
'''
{text}
'''

Given this new part of the document, refine the summary as logically indented markdown bullet points. If the new part is not worth it, simply return the original summary.

SUMMARY OF NEW INFORMATIONS IN MARKDOWN AND IN THE SAME LANGUAGE:
"""
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)
