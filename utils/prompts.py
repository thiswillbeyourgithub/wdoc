from langchain import PromptTemplate, LLMChain

prompt_template = """Write a summary of the this text.
You have to summarize each information of the input text as markdown bullet points.
You have to organize the informations hierarchically using indentation.
You have to write the summary in the same language as the input text.

'''
{text}
'''

SUMMARY AS LOGICALLY INDENTED MARKDOWN BULLET POINTS:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    """Your job is to continue writing a summary of a text.
    You have to summarize each new information of the supplied text as markdown bullet points.
    You have to organize the informations hierarchically using indentation.
    You have to write the summary in the same language as the input text.

    We have provided an existing summary up to this point:
    '''
    {existing_answer}
    '''

    You have to continue the summary by adding the indented bullet points of the following part of the article (only if relevant, stay concise, avoid expliciting what is implied by the previous bullet points):
    '''
    {text}
    '''
    Given this new section of the document, refine the summary as logically indented markdown bullet points. If the new section is not worth it, simply return the original summary."""
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)
