from textwrap import dedent
from langchain import PromptTemplate

summary_rules = dedent("""
- Regarding the formatting of your answer:
    - Use markdown bullet points to format your answer.
    - Use indentation to organize information hierarchically.
    - The bullet points for each information have to follow the order of appearance in the text.
    - The present rules are a good example of adequate formatting.
- Regarding the content of your answer:
    - Include all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, etc.
    - Exclude sponsors, advertisements, embellishments, headers, etc.
    - When in doubt about wether to include an information or not, include it.
    - Direct quotations are allowed.
    - Your answer can contain many bullet points but each one has to be brief and to the point. You don't have to use complete sentences.
        - Good example of brevity : US President's daily staff memo, the present rules.
        - Avoid repetitions, use pronouns if the subject is implied
        - Use common words but keep technical details if noteworthy.
    - Write in the same language as the text: if the text is in French, write an answer in French.
    - Write without bias and stay faithful to the author.
    - If additional information about the text is given (e.g. title): your answer must contain a satisfactory statement about it. i.e. if the text is clickbaity or is a question, add a bullet point that answers the title.
        - In that case: your bullet point must start with "- TITLE EXPLAINER:" and appear first in your answer.
    - You are allowed to judge the author to give a feel of their state of mind.
        - In that case: start with "- JUDGEMENT:" and put this bullet point last.
    - If the text is followed by comments to the article: include them in your answer as if part of the text (but mention that you're now summarizing comments).
""".strip())

prompt_template = """Your job is to summarize a text while following some rules. The text is very long so we'll first give you only the first part and you will later have the right to refine it.

Here are the rules:
'''
{rules}
'''

{metadata}
Here's the first part of the text:
'''
{text}
'''

Here's a reminder of the rules:
'''
{rules}
'''

ANSWER:"""
summarize_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "metadata", "rules"])

refine_template = (
        """You are given two texts: 1. a summary of the first part of a text and 2. the next part of the text. Your job is to update the summary with any new information from the new part of the text while following some rules.

{metadata}
Here's the summary of the text until the new section:
'''
{existing_answer}
'''

Here's the new section of the text:
'''
{text}
'''

Here are the rules to follow while updating the summary:
'''
{rules}
- Don't forget any information from the previous summary: you can reformulate to compress information but cannot omit it.
- If no changes are needed, simply answer the previous summary without any change.
'''

Given this new section of the text and the rules, update the previous summary.

UPDATED SUMMARY:"""
)
refine_prompt = PromptTemplate(
    template=refine_template,
    input_variables=["existing_answer", "text", "metadata", "rules"],
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
