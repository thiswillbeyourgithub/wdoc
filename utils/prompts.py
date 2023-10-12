from textwrap import dedent, indent

# rule for the summarization
summary_rules = dedent("""
- Summary format
  - Use markdown bullet points, with indentation
  - Maintain the rough chronological order of the text
  - You can use direct quotations
  - If the text is in french, reply in french. otherwise reply in english
  - Write in a journalistic tone to stay faithful to the author
  - Keep sentences short and simple
      - Good example of brevity: us president's daily staff memo, the present rules
      - Avoid redundancies / repetitions, especially for sentence subjects
- Summary content
  - What to keep
      - Keep all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc
      - Ignore sponsors, advertisements, etc
      - When in doubt about an information, keep it
      - Use as many bullet points as you need
""".strip())

# template to summarize
system_summary_template = dedent("""You are my best assistant. Your job is to summarize a section of text.

- Rules you absolutely have to follow
{indent(rules, 2)}

""".strip())

human_summary_template = dedent("""{metadata}{previous_summary}

Text section:
'''
{text}
'''

Summary:
""".strip())


# templates to make sure the summary follows the rules
checksummary_rules = dedent("""
- Remove redundancies like "he says that" "he mentions that" etc and use indentation instead because it's implied
- Remove repetitions, especially for pronouns and use implicit reference instead
- Reformulate every bullet point to make it concise but without losing meaning
- Don't use complete sentences
- Use indentation to hierarchically organize the summary
- Don't translate the summary. if the input summary is in french, answer in french
- If the summary is already good, simply answer the same unmodified summary
- Don't omit any information from the input summary in your answer
- A formatted summary that is too long is better than a formatted summary that is missing information from the original summary
""".strip())

system_checksummary_template = dedent("""You are my best assistant. Your job is to fix the format of a summary.

- Rules you absolutely have to follow:
{indent(rules, 2)}
""".strip())

human_checksummary_template = dedent("""Summary to format:
'''
{summary_to_check}
'''

Formatted summary:
""".strip())
