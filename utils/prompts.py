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
\t\t- Use ONE bullet point per sentence. Use indentation to make it easily readable.
\t\t- Write in LANGUAGE.
\t\t- Reformulate direct quotes to be concise whilst keeping the idea and tone of the author.
\t\t- Highlight keywords using bold: e.g. **keyword**.
\t\t- Avoid repetitions:  e.g. don't start several bullet points by 'The author thinks that', just say it once then use indentation to make it implied.
""".strip()), "\t")

# template to summarize
system_summary_template = dedent("""You are my best assistant. I give you a section of a text for you to summarize. What I want is to know the thought process of the authors, their arguments etc and not just high level takeaways. Note that after the whole text has been summarized, I sometime give it back to you to further increase the quality so be careful not to omit information I would want to read!

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
