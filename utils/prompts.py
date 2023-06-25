from langchain import PromptTemplate, LLMChain

summary_rules = """
- Here are the rules:
    - Regarding the formatting of your answer:
        - You have to use markdown bullet points to format your answer.
        - The bullet points have to be indented to organize information hierarchically.
        - These rules are a good example of proper formatting.
    - Regarding the content of your answer:
        - All noteworthy information, anecdotes, facts, insights, etc. must appear in your answer.
        - Information to be discarded include for example sponsors, advertisements, embellishments, headers, etc.
        - Information has to appear in a similar order as chronologically appearing in the text.
        - If relevant, you can use direct quotations from the text.
        - Your answer can be long but has to be brief, quick to skim, to the point, and compressed. You don't have to use full sentences.
            - A fine example of brevity is the daily staff memo to the US President.
            - Avoid repetitions, you can use pronouns, especially if subject sentence is implied.
            - To facilitate reading use common words but keep technical details if noteworthy.
        - Your answer has to be written in the same language as the text: if the text is in French, write an answer in French.
        - Your answer has to be unbiased and faithful to the author.
    - Sometimes, additional information about the text may be given to you such as a title. As it is usually the reason why this text was given to you, your answer also has to contain a satisfactory statement about it. i.e. if the text is clickbaity or a question, add a bullet point that answers it.
        - In that case, this statement must start with "- TITLE EXPLAINER:" and appear as the first bullet point in your answer.
"""

prompt_template = """Your job is to condense a text section by section while following some rules.

{title}
Here's the first section of the text:
'''
{text}
'''

{rules}

Your answer:
"""
summarize_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "title", "rules"])

refine_template = (
    """Your job is to continue condensing a text while following some rules.

{title}
Here's the condensed version so far:
'''
{existing_answer}
'''

Here's the next section of the text:
'''
{text}
'''

{rules}

Given this new section of the text and the rules, refine the condensed version. If no changes are needed, simply answer the condensed text.

Your answer:
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
