<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png"
width="300">

# Construct a QA Bot that Leverages LangChain and LLMs to Answer Questions from Loaded Documents

In this project, you will construct a question-answering (QA) bot. This bot will leverage LangChain and a large language model (LLM) to answer questions based on content from loaded PDF documents. To build a fully functional QA system, you\'ll combine various components, including document loaders, text splitters, embedding models, vector databases, retrievers, and Gradio as the front-end interface.

Imagine you\'re tasked with creating an intelligent assistant that can quickly and accurately respond to queries based on a company\'s extensive library of PDF documents. This could be anything from legal documents to technical manuals. Manually searching through these documents would be time-consuming and inefficient.


<div style="text-align: center;">
	<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQt6Me37wFKPDtXtylFLFA/qabot.png" alt="Alt text" width="70%" height="auto">
	<p>Source: DALL-E</p>
</div>

In this project, you will construct a QA bot that automates this process. By leveraging LangChain and an LLM, the bot will read and understand the content of loaded PDF documents, enabling it to provide precise answers to user queries. You will integrate the tools and techniques, from document loading, text splitting, embedding, vector storage, and retrieval, to create a seamless and user-friendly experience via a Gradio interface.

## Learning objectives

By the end of this project, you will be able to:

- Combine multiple components, such as document loaders, text splitters, embedding models, and vector databases, to construct a fully functional QA bot
- Leverage LangChain and LLMs to solve the problem of retrieving and answering questions based on content from large PDF documents

## Setting up a virtual environment

Let\'s create a virtual environment. Using a virtual environment allows you to manage dependencies for different projects separately, avoiding conflicts between package versions.

In the terminal of your Cloud IDE, ensure that you are in the path `/home/project`, then run the following commands to create a Python virtual environment.

```shell
pip install virtualenv
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env
```

## Installing necessary libraries

To ensure seamless execution of your scripts, and considering that certain functions within these scripts rely on external libraries, it\'s essential to install some prerequisite libraries before you begin. For this project, the key libraries you\'ll need are `Gradio` for creating user-friendly web interfaces and `IBM-watsonx-AI` for leveraging advanced LLM models from the IBM watsonx API.

- **[`gradio`](https://www.gradio.app/)** allows you to build interactive web applications quickly, making your AI models accessible to users with ease.
- **[`ibm-watsonx-ai`](https://ibm.github.io/watsonx-ai-python-sdk/)** for using LLMs from IBM watsonx.ai.
- **[`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/)** for using relevant features from Langchain.
- **[`chromadb`](https://www.trychroma.com/)** for using the chroma database as a vector database.
- **[`pypdf`](https://pypi.org/project/pypdf/)** is required for loading PDF documents.

Here\'s how to install these packages (from your terminal):

```shell
# installing necessary packages in my_env
python3.11 -m pip install \
gradio==4.44.0 \
ibm-watsonx-ai==1.1.2  \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11 \
chromadb==0.4.24 \
pypdf==4.3.1 \
pydantic==2.9.1
```

Now, the environment is ready to create the application.

## Construct the QA bot

It\'s time to construct the QA bot!

## Import the necessary libraries

Inside `qabot.py`, import the following from `gradio`, `ibm_watsonx.ai`, `langchain_ibm`, `langchain`, and `langchain_community`. The imported classes are necessary for initializing models with the correct credentials, splitting text, initializing a vector store, loading PDFs, generating a question-answer retriever, and using Gradio.

```python
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr
```


## Initialize the LLM

```python
## LLM
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm
```

## Define the PDF document loader

```python
## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document
```
## Define the text splitter

```python
## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks
```
## Define the embedding model

```python
## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding
```
## Define the vector store

```python
## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb
```

## Define the retriever

```python
## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever
```
## Define a question-answering chain

```python
## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']
```

Let\'s recap how all the elements in our bot are linked. 

- Note that `RetrievalQA` accepts an LLM (`get_llm()`) and a retriever object (an instance generated by `retriever()`) as arguments. 
- However, the retriever is based on the vector store (`vector_database()`), which in turn needed an embeddings model (`watsonx_embedding()`) and chunks generated using a text splitter (`text_splitter()`). 
- The text splitter, in turn, needed raw text, and this text was loaded from a PDF using `PyPDFLoader`. 
- This effectively defines the core functionality of your QA bot!


## Set up the Gradio interface

Given that you have created the core functionality of the bot, the final item to define is the Gradio interface. [Click Here](https://www.coursera.org/learn/project-generative-ai-applications-with-rag-and-langchain/ungradedLti/sysQs/lab-set-up-a-simple-gradio-interface-to-interact-with-your-models "Click Here") to refer the lab. Your Gradio interface should include:
- A file upload functionality (provided by the `File` class in Gradio)
- An input textbox where the question can be asked (provided by the `Textbox` class in Gradio)
- An output textbox where the question can be answered (provided by the `Textbox` class in Gradio)

Add the following code to `qabot.py` to add the Gradio interface:

```python
# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File", file_count="single", 
            file_types=['.pdf'], type="filepath"),
        gr.Textbox(
            label="Input Query", lines=2, 
            placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Final Project Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)
```



## Add code to launch the application

```python
# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port= 7860)
```

## Author(s)
[Kang Wang](https://author.skills.network/instructors/kang_wang)

[Wojciech \"Victor\" Fulmyk](https://www.linkedin.com/in/wfulmyk)

[Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)

[Ricky Shi](https://author.skills.network/instructors/ricky_shi)


## Contributor(s)
[Hailey Quach](https://author.skills.network/instructors/hailey_quach)

<!--
## Changelog
| Date | Version | Changed by | Change Description |
|------|--------|--------|---------|
| 2024-08-12 | 0.1  | Wojciech "Victor" Fulmyk  | Lab edited; original content written by Kang Wang  |
| 2024-09-16 | 0.2 | Praveen Thapliyal | QA review |
| 2025-03-28 | 0.3 | Alok Narula | ID review |
| 2025-03-28 | 0.4 | Andrea Hansis | Content QA |
| 2025-04-29 | 0.5 |Wojciech "Victor" Fulmyk  | Content QA; update mention of Llama 3.1 to 3.3 |
| 2024-08-12 | 0.6  | Sowmyaa Gurusamy  | Lab edited; As per Final Submission  |

-->


<footer>
<img align="left" src="
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-CD0210EN-Coursera/images/SNIBMfooter.png"
alt="">
</footer>