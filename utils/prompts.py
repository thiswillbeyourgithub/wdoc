from textwrap import dedent, indent

# rule for the summarization
summary_rules = indent(dedent("""
- Summary format
\t- Use markdown bullet points, USE INDENTATION TO SHOW THE LOGIC
\t- Maintain the rough chronological order of the text
\t- If the text is already a summary, make it better but don't remove any information without checking the rules first
\t- You can use direct quotations
\t- If the text is in French, write in French. otherwise reply in English
\t- Write in a journalistic tone to stay faithful to the author
\t- Keep sentences short and simple
\t\t\t- Good example of brevity: us president's daily staff memo, the present rules
\t\t\t- Avoid redundancies / repetitions, especially for sentence subjects
- Summary content
\t- What to keep
\t\t\t- Keep all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc
\t\t\t- Ignore sponsors, advertisements, etc
\t\t\t- When in doubt about an information, keep it
\t\t\t- Use as many bullet points as you need
""".strip()), "\t")

# template to summarize
system_summary_template = dedent("""You are my best assistant. Your job is to summarize a section of text.

- Rules
{rules}

""".strip())

human_summary_template = dedent("""{metadata}{previous_summary}

Text section:
'''
{text}
'''

Summary:
""".strip())


# templates to make sure the summary follows the rules
checksummary_rules = indent(dedent("""
- Remove redundancies like "he says that" "he mentions that" etc and use indentation instead because it's implied
- Remove repetitions, especially for pronouns and use implicit reference instead
- Reformulate every bullet point to make it concise but without losing meaning
- Don't use complete sentences
- Use indentation to hierarchically organize the summary
- Don't translate the summary. if the input summary is in french, answer in french
- If the summary is already good, simply answer the same unmodified summary
- Don't omit any information from the input summary in your answer
- A formatted summary that is too long is better than a formatted summary that is missing information from the original summary
""".strip()), "\t")

system_checksummary_template = dedent("""You are my best assistant. Your job is to fix the format of a summary.

- Rules
{rules}
""".strip())

human_checksummary_template = dedent("""Summary to format:
'''
{summary_to_check}
'''

Formatted summary:
""".strip())
