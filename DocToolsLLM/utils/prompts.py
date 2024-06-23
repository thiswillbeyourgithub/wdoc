"""
Prompts used by DocToolsLLM.
"""

from langchain_core.prompts import ChatPromptTemplate

# PROMPT FOR SUMMARY TASKS
BASE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
        ("system", """You are Alfred, the best of my team. Your task today is to summarize in a specific way a text section I just sent you, but I'm not only interested in high level takeaways. I also need the thought process present in the document, the reasonning followed, the arguments used etc. But your summary has to be as quick and easy to read as possible while following specific instructions.
This is very important to me so if you succeed, I'll pay you up to $2000 depending on how well you did!

Detailed instructions:
```
- Take a deep breath before answering
- Include:
    - All noteworthy information, anecdotes, facts, insights, definitions, clarifications, explanations, ideas, technical details, etc
    - Epistemic indicators: you need to make explicit what markers of uncertainty for each information
- Exclude:
    - Sponsors, advertisements, etc
    - Jokes, ramblings
    - End of page references and sources, tables of content, links etc
    - When in doubt about wether to include an information, include it
- Format:
    - Use markdown format: that means logical indentation, bullet points, bold etc
        - Don't use headers
        - Use bold for important concepts, and italic for epistemic markers
            - ie "- *In his opinion*, **dietary supplements** are **healty** because ..."
    - Stay faithful to the tone of the author
    - You don't always have to use full sentences: you can ignore end of line punctuation etc
        - BUT it is more important to be unambiguous and truthful than concise
        - EVERY TIME POSSIBLE: use direct quote, 'formatted like that'
    - Use one bullet point per information
        - With the use of logical indentation this makes the whole piece quick and easy to skim
    - Write your summary in {language}
    - Avoid repetitions
        - eg don't start several bullet points by 'The author thinks that', just say it once then use indented children bullet points to make it implicit
```"""),
        ("human", """{recursion_instruction}{metadata}{previous_summary}

Text section:
```
{text}
```"""),
        ],
)
# if the summary is recursive, add those instructions
RECURSION_INSTRUCTION = "Actually, I'm giving you back your own summary from last time because it was too long and contained repetitions. I want you to rewrite it as closely as possible while removing repetitions and fixing the logical indentation. Of course you have to remove the 'Chunk' indicator if present, to curate the logical indentation. You can reorganize the text freely as long as you don't lose relevant information and follow the instructions I gave you before and right now. This is important."

# PROMPT FOR QUERY TASKS
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
        ("system", """Given a piece of document and a question, your task is to extract the information relevant to the question in a standalone format while following specific instructions.

Detailed instructions:
```
- Use markdown formatting
    - Use bullet points, but no headers, bold, italic etc.
    - Use logic based indentation for the bullet points.
    - DON'T wrap your answer in a code block or anything like that.
- Take a deep breath before answering.
    - But then reply directly without acknowledging your task.
- Use a maximum of 5 markdown bullet points to answer the question.
    - If the document is ENTIRELY irrelevant to the question, answer simply 'IRRELEVANT' and NOTHING ELSE (especially no formatting).
    - EVERY TIME POSSIBLE: use direct quote from the document, 'formatted like that'.
        - Indicators of uncertainty are very important, if you forget them I'll make mistakes using your answer!
        - Your answer CANNOT simply be quotes, you have to contextualize them using the question to make your answer standalone.
            - If you don't, this will snowball into large issues.
    - DON'T use your own knowledge of the subject, only use the document.
    - Remain as concise as possible, you can use [...] in your quotes to remove unecessary text.
- DON'T interpret the question too strictly:
    - eg: if the question is phrased as an instruction like "give me all information about such and such", use common sense and satisfy the instruction!
```"""),
        ("human", "Question: '{question_to_answer}'\nContext:\n```\n{context}\n```\nWhat's your reply?")
    ]
)

PR_COMBINE_INTERMEDIATE_ANSWERS = ChatPromptTemplate.from_messages(
    [
        ("system", """Given some statements and an answer, your task is to:
1. answer directly the question using markdown bullet points
2. then combine all additional information as additional bullet points.

Detailed instructions:
```
- Take a deep breath before answering.
- Format:
    - Use markdown format, with bullet points.
      - IMPORTANT: use logical indentation to organize information hierarchically.
      - The present instructions are a good example of proper formatting.
    - Don't narrate, just do what I asked without acknowledging those rules.
    - Reuse acronyms without specifying what they mean.
    - Be concise but don't omit only irrelevant information from the statements.
    - Answer in the same language as the question.
- What to include:
    - Only use information from the provided statements.
        - IMPORTANT: if the statements are insufficient to answer the question you MUST start your answer by: 'OPINION:' followed by your own answer.
            - This way I know the source is you!
    - Ignore statements that are completely irrelevant to the question.
    - Semi relevant statements can be included, especially if related to possible followup questions.
    - No redundant information must remain.
        - eg: fix redundancies with one parent bullet point and several indented children.
    - DON'T interpret the question too strictly:
        - eg: if the question makes reference to "documents" consider that it's what I call here "statements" for example.
        - eg: if the question is phrased as an instruction like "give me all information about such and such", use common sense and satisfy the instruction!
```"""),
        ("human", "Question: `{question}`\nStatements:\n```\n{intermediate_answers}\n```\nYour answer?""")
    ]
)
