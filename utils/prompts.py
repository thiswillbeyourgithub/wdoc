from textwrap import dedent, indent
from langchain_core.prompts import ChatPromptTemplate

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

PR_CONDENSE_QUESTION = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a conversation and an additional follow up question, your task is to rephrase this follow up question as a standalone question, in the same language as it was phrased."),
    ("human", "Conversation:\n```\n{chat_history}\n```\nFollow up question: '{question_for_embedding}'\nWhat's your standalone question reformulation?")
        ]
)

# RAG
PR_EVALUATE_DOC = ChatPromptTemplate.from_messages(
    [
        ("system", "You are given a question and text document. Your task is to answer the digit '1' if the text is semantically related to the question otherwise you answer the digit '0'.\nDon't narrate, don't acknowledge those rules, just answer directly the digit without anything else or any formatting."),
        ("human", "Question: '{q}'\nText document:\n```\n{doc}\n```\n\nWhat's your one-digit answer?")
        ]
)

PR_ANSWER_ONE_DOC = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for question-answering tasks.
You are given a piece of document and a question to answer.
If the document is ENTIRELY irrelevant to the question, answer directly 'IRRELEVANT' without anything else and no other formatting.
Otherwise, use a maximum of 3 md bulletpoints to answer the question using only information from the provided document.
Use markdown formatting for easier reading, but don't wrap your answer in a code block or anything like that: reply instantly without acknowledging those rules.
Doing all that you have to remain VERY concise while remaining truthful to the document content.
But DON'T interpret the question too strictly, e.g. the question can be implicit because phrased as an instruction like "give me all information about such and such", use common sense!"""),
        ("human", "Question: '{question_to_answer}'\nContext:\n```\n{context}\n```\nWhat's your reply?")
    ]
)

PR_COMBINE_INTERMEDIATE_ANSWERS = ChatPromptTemplate.from_messages(
    [
        ("system", """Given some statements, your task it to answer a given question using only information from the statements.
Ignore irrelevant statements. Don't narrate, just do what I asked.
Use markdown formatting, especially bullet points for enumeration, bold, indentation etc.
Be VERY concise but don't omit ANY relevant information from the statements.
Answer in the same language as the question.
Above all: if the statements are not enough to answer the question you MUST start your answer by: 'OPINION:' followed by your answer using your own knowledge to let me know the source is you!
But DON'T interpret the question too strictly, for example if the question makes reference to "documents" consider that it's what I call here "statements" for example.
Also the question can for example be an instruction like "give me all information about such and such", use common sense and don't be too strict!
But DON'T interpret the question too strictly, e.g. the question can be implicit because phrased as an instruction like "give me all information about such and such", use common sense!"""),
        ("human", "Question: `{question}`\nStatements:\n```\n{intermediate_answers}\n```\nYour answer?""")
    ]
)
