from textwrap import dedent, indent

# rule for the summarization
summary_rules = """
- Include:
\t- All noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc.
- Exclude:
\t- Sponsors, advertisements, etc.
\t- Jokes, ramblings.
\t- When in doubt, keep the information in your summary.
- Format:
\t- Use markdown format: that means logical indentation, bullet points, bold etc. Don't use headers.
\t- Don't use complete sentence, I'm in a hurry and need bullet points.
\t- Use one bullet point per information, with the use of logical indentation this makes the whole piece quick and easy to skim.
\t- Use bold for important concepts (i.e. "- Mentions that **dietary supplements are healty** because ...")
\t- Write in [LANGUAGE].
\t- Reformulate direct quotes to be concise, but stay faithful to the tone of the author.
\t- Avoid repetitions:  e.g. don't start several bullet points by 'The author thinks that', just say it once then use indentation to make it implied..
""".strip()

# template to summarize
system_summary_template = dedent("""
You are Alfred, my best journalist. Your job today is to summarize in a specific way a text section I just sent you. But I'm not interested simply in high level takeaways, what I'm interested in is the thought process of the authors, their arguments etc. The summary has to be as quick and easy to read as possible while following the rules. This is very important so if you succeed, I'll tip you up to $2000![RECURSION_INSTRUCTION]

- Detailed instructions:
 '''
{rules}
 '''

""")

# if the summary is recursive, add those instructions
system_summary_template_recursive = system_summary_template.replace(
        "[RECURSION_INSTRUCTION]",
        "\nFor this specific job, I'm giving you back your own summary because it was too long and contained repetition. I want you to rewrite it as closely as possible while removing repetitions and fixing the logical indentation. You can rearrange the text freely but don't lose information I'm interested in. Don't forget the instructions I gave you. This is important."
        )

system_summary_template = system_summary_template.replace("[RECURSION_INSTRUCTION]", "")

# summary in a model that is not a chat model
human_summary_template = dedent("""
{metadata}{previous_summary}

Text section:
'''
{text}
'''
""")


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
# """).strip(), "\t")
# 
# system_checksummary_template = dedent("""
# You are my best assistant. Your job is to fix the format of a summary.
# 
# - Rules
# {rules}
# """).strip()
# 
# human_checksummary_template = dedent("""
# Summary to format:
# '''
# {summary_to_check}
# '''
# 
# Formatted summary:
# """).strip()

CONDENSE_QUESTION = dedent("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""")

# RAG
EVALUATE_DOC = """Given the following question and document text, if the text is related to the question you answer '1', otherwise you answer '0'.
Don't narrate, just answer the number.
Question: '{q}'
Document:
```
{doc}
```
Answer:"""

ANSWER_ONE_DOC = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the context is irrelevant, just answer 'IRRELEVANT' and nothing else.
Use three sentences maximum.
Be VERY concise and use markdown formatting for easier reading.
Question: '{question}'
Context:
'''
{context}
'''
Answer:"""

COMBINE_INTERMEDIATE_ANSWERS = """Given the following statements, you must answer a given question.
Ignore irrelevant statements. Don't narrate, just do what I asked.
Use markdown formatting, especially bullet points for enumeration etc.
Be VERY concise but don't omit anything from the answers.
Use the same language as the question.
Above all: if the statements are not useful to answer the question you MUST begin your answer by: 'OPINION:' followed by your answer based on your own knowledge so that I know that the answer is coming from you!

Question: `{question}`
Statements:
```
{intermediate_answers}
```
Answer:"""
