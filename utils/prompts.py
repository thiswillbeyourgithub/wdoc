from textwrap import dedent
from langchain import PromptTemplate

summary_rules = dedent("""
* Regarding the formatting of your summary:
    * Use markdown bullet points.
    * Use indentation to organize information hierarchically.
    * The bullet points for each information have to follow the order of appearance in the text.
    * The present rules are a good example of adequate formatting.
* Regarding the content of your summary:
    * Include all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, etc.
    * Exclude sponsors, advertisements, embellishments, headers, etc.
    * When in doubt about wether to include an information or not, include it.
    * Direct quotations are allowed.
    * Your answer can contain many bullet points but each one has to be brief and to the point. You don't have to use complete sentences.
        * Good example of brevity : US President's daily staff memo, the present rules.
        * Use common words but keep technical details if noteworthy.
        * Don't use pronouns if it's implied by the previous bullet point: write like a technical report.
    * Write in the same language as the input: if the text is in French, write an answer in French.
    * Write without bias and stay faithful to the author.
""".strip())

summarize_template = """Your job is to summarize a chunk of a text while following some rules.

{metadata}
{previous_summary}
Here's a chunk of the text:
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
