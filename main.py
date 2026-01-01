from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import gradio as gr

model = OllamaLLM(model="llama3.2", temperature=0.9)

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

chat_history = []

def answer_question(questions_text):
    if not questions_text.strip():
        return chat_history, "Please type at least one question."

    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
    display_text = ""

    for question in questions:
   
        docs = retriever.invoke(question)

      
        reviews_text = ""
        for i, doc in enumerate(docs, 1):
            reviews_text += f"--- Review {i} ---\n{doc.page_content.strip()}\n\n"

        result = chain.invoke({
            "reviews": reviews_text,
            "question": question
        })

      
        parts = result.split("\n\n")
        formatted_result = ""
        for idx, part in enumerate(parts, 1):
            formatted_result += f"*** Part {idx} ***\n{part.strip()}\n\n"

      
        chat_history.append((question, formatted_result))

       
        display_text += f"\n\n========== Question: {question} ==========\n\n{formatted_result}\n\n"



    return  display_text


gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=5, placeholder="Type one or more questions, each in a new line"),
    outputs=gr.Textbox(lines=40),
    title="Pizza Restaurant QA",
    description="Type multiple questions, each in a new line, and the AI will answer each based on customer reviews."
).launch()
