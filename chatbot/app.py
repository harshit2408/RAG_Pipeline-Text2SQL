#importing necssary libraries

from langchain_community.utilities import SQLDatabase
from langchain_cohere import ChatCohere, CohereEmbeddings #using cohere embeddings and chatcohere moedls for now, can try openai (hopefully a better result with it)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #usin recursive chunking method for better chunks of vector data
from langchain_chroma import Chroma #using chroma moedls for storing vecto data
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough #using runnables moedls
from operator import itemgetter
from langchain_core.prompts import PromptTemplate #used to give prompt
from langchain.chains.sql_database.query import create_sql_query_chain #using dedicated langchain chain for processing sql
import os
from dotenv import load_dotenv #for environment variables
import re
import streamlit as st
import cohere


load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatCohere(model="command-r-plus")

class SQLTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self):
        super().__init__(
            chunk_size=1024,
            chunk_overlap=128,
            separators=[
                "\n\nCREATE TABLE",
                "\n\nALTER TABLE",
                "\n\nINSERT INTO",
                ";\n",
                " "
            ],
            is_separator_regex=False
        )

    def split_documents(self, documents):
        chunks = super().split_documents(documents)
        return self._merge_table_definitions(chunks)

    def _merge_table_definitions(self, chunks):
        combined = []
        current_table = None
        
        for chunk in chunks:
            content = chunk.page_content
            if "CREATE TABLE" in content:
                if current_table:
                    combined.append(current_table)
                current_table = chunk
            elif current_table:
                current_table.page_content += "\n" + content
            else:
                combined.append(chunk)
        
        if current_table:
            combined.append(current_table)
            
        return combined

def process_schema(db):
    ddls = []
    relationships = {}
    
    for table_name in db.get_usable_table_names():
        ddl = db.get_table_info([table_name])
        
        # Extract relationships and table properties
        references = re.findall(r'REFERENCES\s+"?(\w+)"?', ddl)
        columns = re.findall(r'^\s*"?(\w+)"?\s+', ddl, re.MULTILINE)
        
        # Ensure all metadata fields are present
        chunk = Document(
            page_content=ddl,
            metadata={
                "object_type": "table_definition",
                "table_name": table_name,
                "related_tables": ", ".join(references),
                "is_central_table": int(len(references) > 2),
                "column_count": len(columns),
                "is_fact_table": int("id" in [col.lower() for col in columns])
            }
        )
        
        # Filter metadata for Chroma compatibility
        chunk.metadata = {k: v for k, v in chunk.metadata.items() 
                         if isinstance(v, (str, int, float, bool))}
        
        ddls.append(chunk)
        relationships[table_name] = references
    
    return ddls, relationships

class SchemaRetriever:
    def __init__(self, vector_store, relationship_graph):
        self.vector_store = vector_store
        self.relationship_graph = relationship_graph
        self.base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

    def find_related_tables(self, table_name):
        # Get documents with metadata from Chroma
        collection = self.vector_store._collection.get()
        related = []
        
        for doc, metadata in zip(collection['documents'], collection['metadatas']):
            if metadata["table_name"] in self.relationship_graph.get(table_name, []):
                related.append(Document(
                    page_content=doc,
                    metadata=metadata
                ))
        
        #print(f"Found related tables for {table_name}: {[r.metadata['table_name'] for r in related]}")  # Debug
        return related

    def get_relevant_documents(self, query):
        #print(f"Processing query: {query}")  # Debug
        
        # Get initial results with scores
        docs_scores = self.vector_store.similarity_search_with_score(query, k=15)
        #print(f"Initial results with scores: {[(doc.metadata['table_name'], score) for doc, score in docs_scores]}")  # Debug
        
        # Adjust score threshold
        filtered_docs = [doc for doc, score in docs_scores if score < 0.7]
        #print(f"Filtered documents: {[doc.metadata['table_name'] for doc in filtered_docs]}")  # Debug
        
        expanded = []
        seen = set()
        
        for doc in filtered_docs:
            tbl = doc.metadata.get("table_name")
            if tbl and tbl not in seen:
                expanded.append(doc)
                seen.add(tbl)
                # Add related tables
                related = self.find_related_tables(tbl)
                expanded += [d for d in related if d.metadata["table_name"] not in seen]
        
        #print(f"Expanded results: {[doc.metadata['table_name'] for doc in expanded]}")  # Debug
        
        # Remove duplicates based on table name to avoid any discrepancy
        unique_results = []
        seen_tables = set()
        for doc in expanded:
            tbl = doc.metadata["table_name"]
            if tbl not in seen_tables:
                unique_results.append(doc)
                seen_tables.add(tbl)
        
        # Sort by importance to allow for better sql query generations by llm
        sorted_results = sorted(unique_results, 
                              key=lambda x: (
                                  -x.metadata.get("is_central_table", 0),
                                  -x.metadata.get("column_count", 0),
                                  -x.metadata.get("is_fact_table", 0)
                              ))[:7]
        
        #print(f"Final sorted results: {[doc.metadata['table_name'] for doc in sorted_results]}")  # Debug
        
        return sorted_results
    
loader = TextLoader("chinook.sql", encoding="utf-8")
documents = loader.load()

splitter = SQLTextSplitter()
chunks = splitter.split_documents(documents)

embedding_model = CohereEmbeddings(model="embed-english-v3.0")
chunks, rel_graph = process_schema(db)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="cohere_schema",
    persist_directory="./chroma_cohere",
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = SchemaRetriever(vector_store, rel_graph)

def create_semantic_sql_chain(llm, db, retriever):
    """Create SQL generation chain with semantic schema retrieval"""
    
    # Formatting function for retrieved documents
    def format_docs(docs):
        formatted = []
        for doc in docs:
            # Include metadata information in the formatted output
            metadata_info = [
                f"Table: {doc.metadata.get('table_name', 'unknown')}",
                f"Columns: {doc.metadata.get('column_count', 'unknown')}",
                f"Relationships: {doc.metadata.get('related_tables', 'none')}"
            ]
            formatted.append(f"{doc.page_content}\nMetadata: {', '.join(metadata_info)}")
        return "\n\n".join(formatted)
    
    template = '''Given an input question, first create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    DO NOT include any markdown formatting, code blocks, or explanation text such as ```sql in start and ``` in end of the SQL query.
    ONLY return the SQL query itself, nothing else.
    The query should be executable and return the results asked in the question.
    You may return up to {top_k} results.

    Only use the following tables and their relationships:

    {table_info}.

    Question: {input}'''
    
    prompt = PromptTemplate.from_template(template)
    # Base SQL query chain
    query_chain = create_sql_query_chain(llm, db, prompt)
    
    # Full chain with semantic retrieval
    return (
        RunnablePassthrough.assign(
            table_info=lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
        )
        | query_chain
    )

## streamlit framework
st.title('Text to SQL Demo With Cohere API')
llm_query=st.text_input("Enter the query u want to answer about:")

chain = create_semantic_sql_chain(llm, db, retriever)
result = chain.invoke({"question": llm_query})

if llm_query:
    import re
    # Clean the result using regex
    cleaned_result = re.sub(r'^```sql\s*|\s*```$', '', result).strip()
    
    st.markdown("**Generated SQL Query:**")
    st.code(cleaned_result, language='sql')
    st.markdown("**Query Results:**")
    st.write(db.run(cleaned_result))