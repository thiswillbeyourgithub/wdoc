from langchain import PromptTemplate, LLMChain
summary_rules = """
Here are the rules:
    - Regarding the formatting of your summary:
        - You have to use markdown bulletpoints to format your summary.
        - The bulletpoints can be indented to help organize information hierarchically.
        - These rules are an example of proper formatting.
    - Regarding the content of your summary:
        - All relevant information, anecdotes, facts, etc must appear in your summary.
        - Your summary has to be written in the same language as the text: if the text is in french, write a summary in french.
        - Information has to appear in roughly the same order as in the text. You can deviate a bit from the original ordering if it helps the hierarchization and helps to compress the summary.
        - Irrelevant information are to be discarded, for example sponsors, advertissement, embellishment, headers etc
        - The summary has to be brief, to the point, compressed.
            - A fine example of brievty is the daily staff memo to the US President.
            - To be quick to read, use as much as possible common words.
        - If relevant, you can use direct quotation form the text.
        - You absolutely have to be truthful and unbiased.
    - Additionally, if the text comes with a title, you must explicitely write a sentence that answers directly to the title. i.e. if the title is a question, answer it. if the title is clickbaity, write a satisfactory explanation etc.
        - In which case, this bullet point must begin with "TITLE EXPLAINER:" and appear as the first bullet point.
"""

prompt_template = """Your job is to write a summary of a text while following some rules.

{title}
Here's the first part of the text:
'''
{text}
'''

[RULES]

SUMMARY IN MARKDOWN:
"""
summarize_prompt = PromptTemplate(
        template=prompt_template.replace("[RULES]", summary_rules),
        input_variables=["text", "title"])
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

[RULES]

Given this new section of the text and the rules, refine the summary. If no changes are needed, simply answer the original summary.

SUMMARY IN MARKDOWN:
"""
)
refine_prompt = PromptTemplate(
    template=refine_template.replace("[RULES]", summary_rules),
    input_variables=["existing_answer", "text", "title"],
)
