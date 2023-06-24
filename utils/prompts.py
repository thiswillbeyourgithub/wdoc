from langchain import PromptTemplate, LLMChain

summary_rules = """
Here are the rules:
    - Regarding the formatting of your summary:
        - You have to use markdown bullet points to format your summary.
        - The bullet points can be indented to help organize information hierarchically.
        - These rules are an example of proper formatting.
    - Regarding the content of your summary:
        - All relevant information, anecdotes, facts, etc. must appear in your summary.
        - Your summary has to be written in the same language as the text: if the text is in French, write a summary in French.
        - Information has to appear in roughly the same order as in the text. You can deviate a bit from the original ordering if it helps with the hierarchization and helps to compress the summary.
        - Irrelevant information is to be discarded, for example sponsors, advertisements, embellishments, headers, etc.
        - The summary has to be brief, to the point, and compressed.
            - A fine example of brevity is the daily staff memo to the US President.
            - To be quick to read, use as much as possible common words.
        - If relevant, you can use direct quotations from the text.
        - You absolutely have to be truthful and unbiased.
        - You can use pronouns if the subject of the sentence is obvious or implied.
    - Additionally, if the text comes with a title, you must explicitly write a sentence that answers directly to the title. i.e. if the title is a question, answer it. If the title is clickbaity, write a satisfactory explanation, etc.
        - In which case, this bullet point must begin with "- TITLE EXPLAINER:" and appear as the first bullet point.
"""

prompt_template = """Your job is to write a summary of a text while following some rules.

{title}
Here's the first part of the text:
'''
{text}
'''

{rules}

SUMMARY IN MARKDOWN:
"""
summarize_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "title", "rules"])

refine_template = (
    """Your job is to continue the summary of a text while following some rules.

{title}
Here's the summary so far:
'''
{existing_answer}
'''

Here's the next section of the text:
'''
{text}
'''

{rules}

Given this new section of the text and the rules, refine the summary. If no changes are needed, simply answer the original summary.

SUMMARY IN MARKDOWN:
"""
)
refine_prompt = PromptTemplate(
    template=refine_template,
    input_variables=["existing_answer", "text", "title", "rules"],
)

# combine_template = """Given the following answer you gave to this question on a long document. Create a final answer.
# If none of your answers were satisfactory, just say that you don't know. But add an extra paragraph starting by "My two cents:" followed by your answer in your own words. Answer in the language the question was asked.
# 
# Original question:
# '''
# {question}
# '''
# 
# Your answers:
# '''
# {answers}
# '''
# 
# Your definitive answer:"""
# combine_prompt = PromptTemplate(
#     template=combine_template,
#     input_variables=["question", "answers"],
# )
# 
# query_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# 
# Context:
# '''
# {context}
# '''
# 
# Question: {question}
# Your answer:"""
# query_prompt = PromptTemplate(
#     template=query_template,
#     input_variables=["context", "question"],
# )
