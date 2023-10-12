from textwrap import dedent, indent

# rule for the summarization
summary_rules = dedent("""
- summary format
  - use markdown bullet points, with indentation
  - maintain the rough chronological order of the text
  - the present rules are a good example of adequate formatting
  - you can use direct quotations
  - if the text is in french, reply in french. otherwise reply in english
  - write without bias and stay faithful to the author
  - good example of brevity: us president's daily staff memo, the present rules
- summary content
  - what to keep
      - keep all noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc
      - ignore sponsors, advertisements, etc
      - when in doubt about an information, keep it
      - use as many bullet points as you need
  - keep sentences short and simple
  - avoid redundancies / repetitions, especially for sentence subjects
""".strip())

# template to summarize
system_summary_template = dedent("""You are my best assistant. Your job is to summarize a section of text.

- RULES YOU ABSOLUTELY HAVE TO FOLLOW
{indent(rules, 2)}

""".strip())

human_summary_template = dedent("""{metadata}{previous_summary}

Text section:
'''
{text}
'''

SUMMARY:
""".strip())


# templates to make sure the summary follows the rules
checksummary_rules = dedent("""
- remove redundancies like "He says that" "He mentions that" etc and use indentation instead because it's implied
- remove repetitions, especially for pronouns and use implicit reference instead
- reformulate every bullet point to make it concise but without losing meaning
- don't use complete sentences
- use indentation to hierarchically organize the summary
- don't translate the summary. If the input summary is in French, answer in French
- if the summary is already good, simply answer the same unmodified summary
- don't omit any information from the input summary in your answer
- a formatted summary that is too long is better than a formatted summary that is missing information from the original summary
""".strip())

system_checksummary_template = dedent("""You are a perfect assistant. Your job is to fix the format of a summary. There are rules you absolutely have to follow.

RULES YOU ABSOLUTELY HAVE TO FOLLOW:
'''
{rules}
'''""".strip())

human_checksummary_template = dedent("""SUMMARY TO FORMAT:
'''
{summary_to_check}
'''

FORMATTED SUMMARY:""".strip())
