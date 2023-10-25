from textwrap import dedent, indent

# rule for the summarization
summary_rules = indent(dedent("""
\t- On the content
\t\t- What to keep
\t\t\t- All noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc.
\t\t- What to discard
\t\t\t- sponsors, advertisements, etc.
\t\t- When in doubt, keep the information in your summary.
\t- On the format
\t\t- Use as many bullet points as you need.
\t\t- The top level bullet points have to follow the order they appear in the text.
\t\t- When summarizing a summary, be careful to FOLLOW THE RULES and to deduplicate lines.
\t\t- Quotations are allowed.
\t\t- Make it quicker to read by showing emphasis (using bold format, e.g. **this is important**).
\t\t- Write your summary IN THE SAME LANGUAGE as the input text.
\t\t- Keep the style and tone of the input text.
\t\t- Keep sentences short and simple, but don't get rid of details.
\t\t- Example of summary: the US president's daily staff memo, also those rules.
\t\t- Avoid redundancies / repetitions, especially for sentence subjects. (E.g. don't begin several bullet points by 'The author says that', just use smart indentation to make it obvious.)
""".strip()), "\t")

# template to summarize
system_summary_template = dedent("""You are my best assistant. I give you a text, section by section and you reply a summary of each section so that I can concatenate them afterwards. I'm not actually interested in the high level take aways, what I want is to know the though process of the authors, what their arguments were etc. To that end I wrote you very specific SUMMARY RULES below. Note that after the whole text has been summarized, I sometime give it back to you to further increase the quality so be careful not to omit information I would want to read!

- SUMMARY RULES
{rules}

""".strip())

human_summary_template = dedent("""{metadata}{previous_summary}

Text section:
'''
{text}
'''

Summary:
""".strip())


# # templates to make sure the summary follows the rules
# checksummary_rules = indent(dedent("""
# - Remove redundancies like "he says that" "he mentions that" etc and use indentation instead because it's implied
# - Remove repetitions, especially for pronouns and use implicit reference instead
# - Reformulate every bullet point to make it concise but without losing meaning
# - Don't use complete sentences
# - Use indentation to hierarchically organize the summary
# - Don't translate the summary. if the input summary is in french, answer in french
# - If the summary is already good, simply answer the same unmodified summary
# - Don't omit any information from the input summary in your answer
# - A formatted summary that is too long is better than a formatted summary that is missing information from the original summary
# """.strip()), "\t")
# 
# system_checksummary_template = dedent("""You are my best assistant. Your job is to fix the format of a summary.
# 
# - Rules
# {rules}
# """.strip())
# 
# human_checksummary_template = dedent("""Summary to format:
# '''
# {summary_to_check}
# '''
# 
# Formatted summary:
# """.strip())
