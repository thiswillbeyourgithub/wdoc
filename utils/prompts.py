from textwrap import dedent
from langchain import PromptTemplate

summary_rules = dedent("""
- Regarding the formatting of your summary:
    - Use markdown bullet points.
    - Use indentation to organize information hierarchically.
    - The bullet points for each information have to follow the order of appearance in the text.
    - The present rules are a good example of adequate formatting.
- Regarding the content of your summary:
    - Include all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, etc.
    - Exclude sponsors, advertisements, embellishments, headers, etc.
    - When in doubt about wether to include an information or not, include it.
    - Direct quotations are allowed.
    - Your answer can contain many bullet points but each one has to be brief and to the point. You don't have to use complete sentences.
        - Good example of brevity : US President's daily staff memo, the present rules.
        - Use common words but keep technical details if noteworthy.
        - Don't use pronouns if it's implied by the previous bullet point: write like a technical report.
    - Write in the same language as the input: if the text is in French, write an answer in French.
    - Write without bias and stay faithful to the author.
""".strip())
#     - If additional information about the text is given (e.g. title): your answer must contain a satisfactory statement about it. i.e. if the text is clickbaity or is a question, add a bullet point that answers the title.
#         - In that case: your bullet point must start with "- TITLE EXPLAINER:" and appear first in your answer.
#     - You are allowed to judge the author to give a feel of their state of mind.
#         - In that case: start with "- JUDGEMENT:" and put this bullet point last.
#     - If the text is followed by comments to the article: include them in your answer as if part of the text (but mention that you're now summarizing comments).

summarize_template = """Your job is to summarize a part of a text while following some rules.

{metadata}
{previous_summary}
Here's a part of the text:
'''
{text}
'''

Here are the rules you have to follow:
'''
{rules}
'''

MARKDOWN SUMMARY:"""
summarize_prompt = PromptTemplate(
        template=summarize_template,
        input_variables=["text", "previous_summary", "metadata", "rules"])

reduce_summaries_template = (
        """You are given a summary of a text. Your job is to see if you can make the summary a bit shorter and faster to read without losing any information and while following some specific rules.

{metadata}
Here's the summary of the text:
'''
{text}
'''

Here are the rules you have to follow:
'''
- Keep using markdown bullet points in your summary.
- Keep as much of the original summary as you can. Modifications can only be used to reformulate bullet points but without losing information.
- You can split a bullet points into sub bullet points as well as use indentation.
- The goal is not to get the key points of the summary but to capture the reasonning, thought process and arguments of the author.
'''

Now given the summary and the rules, update the summary. Keep in mind that I prefer a summary that is too long rather than losing information.

UPDATED SUMMARY:"""
)
reduce_summaries_prompt = PromptTemplate(
    template=reduce_summaries_template,
    input_variables=["text", "metadata"],
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
