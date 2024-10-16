import psycopg2
import numpy as np
import requests
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.llms import ChatOpenAI, AzureChatOpenAI
from langchain.vectorstores import PGVector
from langchain.document_loaders import Document, RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent_with_tools
from langchain.tools import Tool
from typing import List, Optional, Callable, Iterator, Sequence, Union
import logging
import os

# Constants
CONNECTION_STRING = "your_connection_string"  # Replace with your actual connection string
DATABASE_NAME_HISTORY = "your_history_database_name"
DATABASE_HOST = "your_database_host"
DATABASE_USER = "your_database_user"
DATABASE_PASSWORD = "your_database_password"
DATABASE_PORT = "your_database_port"
DATABASE_NAME_EMBEDDINGS = "your_embeddings_database_name"

# Chat functions

def chat_langchain(new_project_qa, query, not_uuid):
    check = query.lower()
    embedding = AzureOpenAIEmbeddings()  # Use Azure embeddings
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=not_uuid,
        embedding_function=embedding,
    )
    project_instance = Project.objects.get(id=not_uuid)
    user_exp_obj = ProjectSetting.objects.get(project=project_instance)
    gpt_key = user_exp_obj.is_gpt_response_enabled

    docs = vector_store.similarity_search(query, k=3)
    if gpt_key:
        prompt = PromptTemplate.from_template('{query}')
        llm1 = AzureChatOpenAI(model='gpt-4o-mini', temperature=0.5)  # Use Azure Chat LLM
        llm_chain = LLMChain(prompt=prompt, llm=llm1)
        retriever = vector_store.as_retriever(search_kwargs={'k': 4})
        relevant_document = retriever.get_relevant_documents(query)
        context_text = '\n\n---\n\n'.join([doc.page_content for doc in relevant_document])
        agent, tools = initialize_agent_with_tools(context_text, new_project_qa, llm_chain)

        try:
            ans = agent.invoke({'input': check, 'tool_names': tools})
            ans_output = ans['output']
            flag = 'true'
        except Exception as e:
            ans = new_project_qa.invoke(query)
            ans_output = ans.get('answer', ans.get('result'))
            flag = 'false'
    else:
        ans = new_project_qa.invoke(query)
        print(ans)
        try:
            ans_output = ans['answer']
        except Exception as e:
            ans_output = ans['result']
        flag = 'false'

    if flag == 'true':
        try:
            intermediate_steps = ans['intermediate_steps']
            source_list = []
            source_docs = intermediate_steps[0][1]['source_documents']
            for doc in source_docs:
                if doc.metadata.get('source') not in source_list:
                    source_list.append(doc.metadata['source'])
        except Exception:
            source_list = ['']
    else:
        try:
            source = docs[0].metadata.get('source', '')
            source_list = [source]
        except Exception:
            source_list = ['']

    relevant_document = source_list
    source = relevant_document[0] if relevant_document else ''
    bot_ending = user_exp_obj.chatbot_ending_msg or ''
    data_source_id = ans['source_documents'][0].metadata['data_source_id']
    list_json = {
        'bot_message': f'{ans_output}\n\n{bot_ending}',
        'citation': source,
        'data_source_id': data_source_id,
    }

    return list_json


def greeting(new_project_qa, query, not_uuid):
    ans = new_project_qa.invoke(query)
    try:
        ans_output = ans['answer']
    except Exception:
        ans_output = ans['result']
    source = ''
    project_instance = Project.objects.get(id=not_uuid)
    project_setting_inst = ProjectSetting.objects.get(project=project_instance)
    bot_ending = (
        project_setting_inst.chatbot_ending_msg
        if project_setting_inst.chatbot_ending_msg is not None
        else ''
    )
    if bot_ending != '':
        list_json = {
            'bot_message': ans_output + '\n\n' + str(bot_ending),
            'citation': source,
        }
    else:
        list_json = {
            'bot_message': ans_output + str(bot_ending),
            'citation': source,
        }
    return list_json


class MultiUrlRecursiveLoader(RecursiveUrlLoader):
    def __init__(
        self,
        urls: List[str],
        max_depth: Optional[int] = 2,
        use_async: Optional[bool] = None,
        extractor: Optional[Callable[[str], str]] = None,
        metadata_extractor: Optional[Callable[[str, str, requests.Response], dict]] = None,
        exclude_dirs: Optional[Sequence[str]] = (),
        timeout: Optional[int] = 10,
        prevent_outside: bool = True,
        link_regex: Union[str, re.Pattern, None] = None,
        headers: Optional[dict] = None,
        check_response_status: bool = False,
        continue_on_failure: bool = True,
    ) -> None:
        self.urls = urls
        super().__init__(
            url=urls[0],
            max_depth=max_depth,
            use_async=use_async,
            extractor=extractor,
            metadata_extractor=metadata_extractor,
            exclude_dirs=exclude_dirs,
            timeout=timeout,
            prevent_outside=prevent_outside,
            link_regex=link_regex,
            headers=headers,
            check_response_status=check_response_status,
            continue_on_failure=continue_on_failure,
        )

    def lazy_load(self) -> Iterator[Document]:
        visited: Set[str] = set()
        results = []
        for url in self.urls:
            if self.use_async:
                results.extend(asyncio.run(self._async_get_child_links_recursive(url, visited)))
            else:
                results.extend(self._get_child_links_recursive(url, visited))
        return iter(results or [])


def generate_embeddings(
    config: dict = None, urls=None, file_path=None, persist_directory=None, data_source=None
):
    encoding = tiktoken.get_encoding('cl100k_base')
    total_tokens = 0
    texts, doc_texts_counters = None, None
    image_types = ['jpeg', 'jpg', 'png', 'gif']

    # Helper function to load and split text
    def load_and_split_documents(loader, chunk_size=None, chunk_overlap=None):
        document = loader.load()
        for doc in document:
            doc.metadata['data_source_id'] = str(data_source.id)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents=document)

    # Helper function for tokenizing and counting tokens
    def tokenize_and_count(texts):
        total_tokens_local = 0
        for text_chunk in texts:
            if isinstance(text_chunk, str):
                tokens = encoding.encode(text_chunk)
                total_tokens_local += len(tokens)
        return total_tokens_local

    # Processing documents
    if file_path:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.pdf':
            texts = load_and_split_documents(
                PDFPlumberLoader(file_path), chunk_size=2000, chunk_overlap=200
            )

        elif file_extension == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8', csv_args={'delimiter': ','})
            texts = load_and_split_documents(loader, chunk_size=2000, chunk_overlap=100)

        elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
            loader = UnstructuredExcelLoader(file_path, mode='elements')
            texts = loader.load()

        elif file_extension in ['.docx', '.doc']:
            texts = load_and_split_documents(
                UnstructuredWordDocumentLoader(file_path), chunk_size=2000, chunk_overlap=100
            )

        elif file_extension in ['.ppt', '.pptx']:
            texts = load_and_split_documents(UnstructuredPowerPointLoader(file_path), chunk_size=3000, chunk_overlap=100)
            print(texts)

        elif any(file_path.lower().endswith(f'.{img_type}') for img_type in image_types):
            texts = load_and_split_documents(
                UnstructuredImageLoader(file_path), chunk_size=3000, chunk_overlap=100
            )

        elif file_extension == '.txt':
            texts = load_and_split_documents(
                TextLoader(file_path), chunk_size=2000, chunk_overlap=100
            )

        doc_texts_counters = texts

    # Process by config (Confluence)
    elif config is not None:
        confluence_url = config.get('confluence_url')
        username = config.get('username')
        api_key = config.get('api_key')
        space_key = list(config.get('space_key', []))
        documents, embedding = [], AzureOpenAIEmbeddings()  # Use Azure embeddings
        loader = ConfluenceLoader(url=confluence_url, username=username, api_key=api_key)
        for space_key in space_key:
            try:
                if not documents:
                    documents = loader.load(space_key=space_key)
            except Exception as e:
                logging.info(f"Error loading confluence space key {space_key}: {str(e)}")
        texts = documents

    # Process by URLs
    elif urls:
        loader = MultiUrlRecursiveLoader(urls)
        texts = loader.lazy_load()

    embedding = AzureOpenAIEmbeddings()  # Use Azure embeddings

    # Calculate total tokens
    total_tokens = tokenize_and_count(texts)
    print("Total tokens:", total_tokens)

    # Initialize vector store
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name='your_collection_name',  # Replace with your collection name
        embedding_function=embedding,
    )

    # Add embeddings to vector store
    PGVector.from_documents(embedding=embedding, documents=texts, collection_name='your_collection_name', connection_string=CONNECTION_STRING)

    return total_tokens


def retrieval_qa_chain(chat_history_session, prompt=None, project_id=None):
    project_instance = Project.objects.get(id=project_id)
    project_setting_inst = ProjectSetting.objects.get(project=project_instance)
    custom_persona = project_setting_inst.custom_persona
    memory_key = project_setting_inst.is_memorization_enabled
    embedding = AzureOpenAIEmbeddings()  # Use Azure embeddings
    llm = AzureChatOpenAI(model='gpt-4o-mini', temperature=0.5)  # Use Azure Chat LLM

    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=project_id,
        embedding_function=embedding,
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


def delete_projects(collection_name):
    embedding = AzureOpenAIEmbeddings()  # Use Azure embeddings
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=collection_name,
        embedding_function=embedding,
    )
    vector_store.delete_collection()


def add_new_data(data, not_uuid):
    embedding = AzureOpenAIEmbeddings()  # Use Azure embeddings
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
