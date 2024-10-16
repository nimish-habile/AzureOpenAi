import os
import psycopg2
import numpy as np
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import PGVector
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

# Initialize Azure OpenAI LLM
def initialize_azure_llm(model_name="gpt-35-turbo", temperature=0.5):
    llm = AzureChatOpenAI(
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        model=model_name,
        temperature=temperature
    )
    return llm

# Initialize Azure OpenAI Embeddings
def initialize_azure_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_version=AZURE_OPENAI_API_VERSION
    )
    return embeddings

# Retrieval QA Chain
def azure_retrieval_qa_chain(chat_history_session, prompt=None, project_id=None):
    project_instance = Project.objects.get(id=project_id)
    project_setting_inst = ProjectSetting.objects.get(project=project_instance)
    custom_persona = project_setting_inst.custom_persona
    memory_key = project_setting_inst.is_memorization_enabled

    llm = initialize_azure_llm()
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=project_id,
        embedding_function=initialize_azure_embeddings(),
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 8})

    message_history = get_message_history_db(chat_history_session, DATABASE_NAME_HISTORY)

    if custom_persona:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            chat_memory=message_history,
        )
        if not memory_key:
            memory.clear()
        chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, memory=memory if memory_key else None
        )
    else:
        if memory_key:
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                chat_memory=message_history,
                output_key='answer',
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={'prompt': prompt},
                rephrase_question=False,
            )
        else:
            chain = RetrievalQA.from_llm(
                llm=llm,
                return_source_documents=True,
                retriever=retriever,
                prompt=prompt,
            )
    return chain

# Add New Data
def add_new_data(data, not_uuid):
    embedding = initialize_azure_embeddings()
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=not_uuid,
        embedding_function=embedding,
    )
    conn = psycopg2.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute('SELECT uuid FROM langchain_pg_collection WHERE name = %s', (not_uuid,))
    document_id = cursor.fetchone()[0]

    cursor.execute(
        'SELECT embedding FROM langchain_pg_embedding WHERE collection_id = %s', (document_id,)
    )
    existing_embedding = cursor.fetchall()
    cursor.execute(
        'SELECT cmetadata FROM langchain_pg_embedding WHERE collection_id = %s', (document_id,)
    )
    data_source_citation = cursor.fetchone()
    citation = data_source_citation[0]['source']
    embeddings_array = np.array([embedding[0] for embedding in existing_embedding])
    metadata = {'source': citation}
    document = Document(page_content=data, metadata=metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents=[document])

    try:
        try:
            PGVector.from_texts(
                embedding=embedding,
                texts=texts,
                collection_name=not_uuid,
                connection_string=CONNECTION_STRING,
            )
        except Exception as e:
            PGVector.from_documents(
                embedding=embedding,
                documents=texts,
                collection_name=not_uuid,
                connection_string=CONNECTION_STRING,
            )
    except Exception as e:
        print(e)
    cursor.close()
    conn.close()
    return True

# Delete Embeddings
def delete_embeddings(file_path, collection_name, data_id=None):
    try:
        data_instance = DataSource.objects.get(id=data_id)

        base_query = f"""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection
            WHERE name = '{collection_name}'
        )
        """

        if data_instance.type == DataSource.Uploads:
            condition = f"AND cmetadata->>'source' = '{file_path}'"

        elif data_instance.type == DataSource.WebURL:
            if data_instance.is_fetch_attachments_enabled:
                condition = f"AND cmetadata->>'source' LIKE '{file_path}%'"
            else:
                condition = f"AND cmetadata->>'source' = '{file_path}'"

        elif data_instance.type == DataSource.Confluence:
            space_key = data_instance.name
            file_path = f'{file_path}/spaces/{space_key}'
            condition = f"AND cmetadata->>'source' LIKE '{file_path}%'"

        sql_query = base_query + condition

        with psycopg2.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            database=DATABASE_NAME_EMBEDDINGS,
            port=DATABASE_PORT,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                conn.commit()

        print('Deleted successfully')

    except Exception as e:
        print(str(e), '----')
