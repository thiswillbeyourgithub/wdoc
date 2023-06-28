from textwrap import dedent

# rule for the summarization
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

# template to summarize
system_summary_template = dedent("""You are a helpful assistant. Your job is to summarize a chunk of a text while following the rules.

Here are the rules you have to follow:
'''
{rules}
'''
""".strip())

human_summary_template = dedent("""{metadata}{previous_summary}

Here's a chunk of the text:
'''
{text}
'''

Here's a reminder of the rules:
'''
{rules}
'''

MARKDOWN SUMMARY:""".strip())


# templates to make sure the summary follows the rules
checksummary_rules = dedent("""
* remove redundancies like "He says that" "He mentions that" etc and use indentation instead because it's implied
* remove repetitions, especially for pronouns and use implicit reference instead
* reformulate every bullet point to make it concise but without losing meaning
* don't use complete sentences
* use indentation to hierarchically organize the summary
* don't translate the summary. If it's in French, answer in French
* if the summary is already good, simply answer the same unmodified summary
* don't omit any information from the input summary in your answer
* a summary that is too long is better than a summary missing information
""".strip())

system_checksummary_template = dedent("""You are a helpful assistant. Your job is to format a summary while following some rules.

RULES:
'''
{rules}
'''""".strip())

human_checksummary_template = dedent("""SUMMARY TO FIX:
'''
{summary_to_check}
'''

RULES REMINDER:
'''
{rules}
'''

FIXED SUMMARY:
""".strip())
