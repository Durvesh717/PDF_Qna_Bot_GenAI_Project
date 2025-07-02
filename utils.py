import streamlit as st
import pymupdf4llm
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from typing import List

def PDF_to_markdown(file_path):
    md_text = pymupdf4llm.to_markdown(
        doc=file_path,  
        page_chunks=True,  
        show_progress=True
    )
    return md_text

def Parse_PDF(file_path):
    loader = UpstageDocumentParseLoader(
                file_path, split="page", 
                output_format="markdown",
                base64_encoding=["figure", "chart", "table"]
            )
    docs = loader.load_and_split()
    return docs


def create_image_descriptions(docs):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    new_documents = []
    
    for doc in docs:
        if 'base64_encodings' in doc.metadata and len(doc.metadata['base64_encodings']) > 0:
            for idx, img_base64 in enumerate(doc.metadata['base64_encodings']):
                message = HumanMessage(
                    content=[
                        {"type": "text", 
                         "text": """
                                    Describe only the factual content visible in the image:

                                    1. If decorative/non-informational: output '<---image--->'

                                    2. For content images:
                                    - General Images: List visible objects, text, and measurable attributes
                                    - Charts/Infographics: State all numerical values and labels present
                                    - Tables: Convert to markdown table format with exact data

                                    Rules:
                                    * Include only directly observable information
                                    * Use original numbers and text without modification
                                    * Avoid any interpretation or analysis
                                    * Preserve all labels and measurements exactly as shown
                                """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
                

                response = model.invoke([message])
                

                new_doc = Document(
                    page_content=response.content,
                    metadata={
                        "page": f"{doc.metadata.get('page', 'unknown')}"
                    }
                )
                
                new_documents.append(new_doc)
    
    return new_documents

def merge_text_and_images(md_text, image_description_docs):

    page_contents = defaultdict(list)
    page_metadata = {}
    

    for text_item in md_text:

        page = int(text_item['metadata']['page'])
        page_contents[page].append(text_item['text'])

        if page not in page_metadata:
            page_metadata[page] = {
                'source': text_item['metadata']['file_path'],
                'page': page
            }
    

    for img_doc in image_description_docs:

        page = int(img_doc.metadata['page'])
        page_contents[page].append(img_doc.page_content)
    
    merged_docs = []
    for page in sorted(page_contents.keys()):
        full_content = '\n\n'.join(page_contents[page])
        doc = Document(
            page_content=full_content,
            metadata=page_metadata[page]
        )
        merged_docs.append(doc)
    
    return merged_docs


def extract_content_from_pdf(file_path):

    try:
        md_text = PDF_to_markdown(file_path)
        docs = Parse_PDF(file_path)
        image_description_docs = create_image_descriptions(docs)
        merged_docs = merge_text_and_images(md_text,image_description_docs)
        return merged_docs
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None



def create_vector_store(merged_docs):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(merged_docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings
            )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None
    

def retrieve_docs(vector_store,question: str) -> List[Document]:
    """Retrieve relevant documents for the given question"""
    print(f"SEARCHING DOCUMENTS...\n{'='*20}")
    retrieved_docs = vector_store.similarity_search(question)
    print(f"searched...{retrieved_docs[0].page_content[:100]}\n...\n{'='*20}")
    return retrieved_docs

def format_docs(docs: List[Document]) -> str:
    """Format documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)



def create_conversation_chain(vector_store):
    """Create conversation chain using the new LangChain architecture"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        prompt = ChatPromptTemplate([
            ("human", """
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Question: {question} 
                Context: {context} 
                Answer:
                """)])
        
        rag_chain = (
            RunnableParallel({
                "context": lambda x: format_docs(retrieve_docs(vector_store,x["question"])),
                "question": RunnablePassthrough()
            })
            | prompt
            | model
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None
