RAG_PROMPT1 = 'You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don\'t know the answer, just say "I don\'t know". \
Use three sentences maximum and keep the answer concise. \
\nQuestion: [question] \nContext: [context] \nAnswer:'

RAG_PROMPT2 = 'Context information is below. Given the context information and not prior knowledge, answer the question. \
\nQuestion: [question] \nContext: [context] \nAnswer:'

RAG_PROMPT3 = 'You are a customer support agent. You are designed to be as helpful as possible while providing only factual information. \
You should be friendly, but not overly chatty. Context information is below. \
Given the context information and not prior knowledge, answer the question. \
\nQuestion: [question] \nContext: [context] \nAnswer:'

MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

OPENENDED_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'


def wrap_prompt(question, context, prompt_id=1) -> str:
    rag_prompts = [RAG_PROMPT1, RAG_PROMPT2, RAG_PROMPT3, MULTIPLE_PROMPT]
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

