import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]
os.environ["USER_AGENT"] = "my-ragas-eval/1.0"

#loading data from webpage
page_url = "https://www.educosys.com/course/genai"
loader = WebBaseLoader(web_paths=[page_url])
docs = loader.load()


#text-splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=80,
)
texts = text_splitter.split_documents(docs)
print(texts)

#Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import  Chroma

vector_db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory="chroma_db")
vector_db.persist()

retriver = vector_db.as_retriever()

#rag query
from langchain import hub
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser

llm = OpenAI()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def print_prompt(prompt):
    print("Prompt - ")
    return prompt

rag_chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(print_prompt)
    | prompt
    | llm
    | StrOutputParser()
)

# response = rag_chain.invoke("what is genai?")
# print(response)

from ragas import *
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Example evaluation questions
eval_questions = [
    "What is GenAI?",
    "Are the recordings of the course available? For how long?",
    "Are the testimonials of the course available? Name the students who have shared testimonials?",
    "Are the certificates for the course provided?"
]

reference_answers = [
    "Generative AI (GenAI) refers to AI systems that can generate content such as text, images, and code.",
    "Yes, the recordings are available with lifetime access.",
    "Yes, testimonials are available. Students include Ashish Upreti, Sathish Krishna, Raman Sharma, Sudarshan Suresh Srikant, Ruthira Sekar, Abhijit Mone, Manika Kaushik.",
    "Yes, certificates are provided for the course."
]

samples = []

for question, reference in zip(eval_questions, reference_answers):
    # Step 1: Retrieve top 3 relevant documents
    retrieved_docs = retriver.invoke(question)[:3]  # top 3 docs
    retrieved_contexts = [doc.page_content for doc in retrieved_docs if doc.page_content]

    # Step 2: Generate answer automatically using RAG
    generated_answer = rag_chain.invoke(question)

    # Step 3: Save sample
    samples.append({
        "question": question,
        "answer": generated_answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference
    })

# Step 4: Convert to Hugging Face Dataset
dataset = Dataset.from_list(samples)


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)


results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)

print(results)