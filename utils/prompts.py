from langchain import PromptTemplate, LLMChain

summary_rules = """
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
    - If the text is followed by comments to the article: include them in your answer as if part of the text (but mention that you're now condensing comments).
"""

prompt_template = """Your job is to condense a text while following some rules. The text is very long so we'll first give you only the first section and you will later have the right to refine it.

Here are the rules:
'''
{rules}
'''

{metadata}
Here's the first section of the text:
'''
{text}
'''

Here's a reminder of the rules:
'''
{rules}
'''

Your answer:
"""
summarize_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "metadata", "rules"])

refine_template = (
    """Your job is to continue condensing a text while following some rules.

{metadata}
Here's the condensed version so far:
'''
{existing_answer}
'''

Here's a reminder of the rules:
'''
{rules}
'''

Here's the next section of the text:
'''
{text}
'''

Given this new section of the text and the rules, refine the condensed version. If no changes are needed, simply answer the condensed text.

Your answer:
"""
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
