#importing necssary libraries

from langchain_community.utilities import SQLDatabase
from langchain_cohere import ChatCohere, CohereEmbeddings #using cohere embeddings and chatcohere moedls for now, can try openai (hopefully a better result with it)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #usin recursive chunking method for better chunks of vector data
from langchain_chroma import Chroma #using chroma moedls for storing vecto data
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import time
import ast

load_dotenv()
#os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#llm = ChatCohere(model="command-r-plus")
llm = ChatOpenAI(model="gpt-4o-mini")

db = SQLDatabase.from_uri(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")


class SQLTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self):
        #print("Initializing SQLTextSplitter with custom separators")
        super().__init__(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "/* Table:", "CREATE TABLE"],  # Split at table definitions
            is_separator_regex=False
        )
        #print(f"Configured with chunk_size=2000, chunk_overlap=200, and {len(self._separators)} separators")

    def split_documents(self, documents):
        #print(f"Starting to split {len(documents)} documents")
        chunks = super().split_documents(documents)
        #print(f"Generated {len(chunks)} initial chunks, merging table definitions")
        return self._merge_table_definitions(chunks)

    def _merge_table_definitions(self, chunks):
        #print(f"Starting to merge {len(chunks)} chunks")
        combined = []
        current_table = None
        current_comments = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            #print(f"\nProcessing chunk {i+1}/{len(chunks)}")
            #print(f"Chunk content preview: {content[:100]}...")  # Show first 100 chars
            
            if content.startswith("--"):
                #print("Found comment, adding to current_comments")
                current_comments.append(content)
            elif "CREATE TABLE" in content:
                if current_table:
                    #print("Found new table definition, finalizing previous table")
                    if current_comments:
                        #print(f"Adding {len(current_comments)} comments to table")
                        current_table.page_content = "\n".join(current_comments) + "\n" + current_table.page_content
                        current_comments = []
                    combined.append(current_table)
                #print("Starting new table definition")
                current_table = chunk
            elif current_table:
                #print("Appending content to current table definition")
                current_table.page_content += "\n" + content
            else:
                #print("Adding standalone chunk to combined list")
                combined.append(chunk)
        
        if current_table:
            if current_comments:
                #print(f"Adding final {len(current_comments)} comments to last table")
                current_table.page_content = "\n".join(current_comments) + "\n" + current_table.page_content
            #print("Adding final table to combined list")
            combined.append(current_table)
            
        #print(f"Finished merging, returning {len(combined)} combined chunks")
        return combined

def process_tables(content):
    """Process SQL content into one chunk per table with full context"""
    # Split content into complete table definitions
    table_defs = re.split(r'(?=CREATE TABLE\b)', content)
    
    chunks = []
    seen_tables = set()
    
    for table_def in table_defs:
        if not table_def.strip():
            continue

        # Extract table name using more precise regex
        table_match = re.search(r'CREATE TABLE (\S+)\s*\(', table_def)
        if not table_match:
            continue
            
        table_name = table_match.group(1)
        if table_name in seen_tables:
            continue
            
        # Extract all JSONB fields
        jsonb_fields = re.findall(r'\b(\w+)\s+JSONB\b', table_def)
        
        # Extract relationships from foreign keys
        relationships = re.findall(r'REFERENCES\s+(\w+)', table_def)
        
        # Extract table comments
        table_comments = re.findall(r'/\*.*?\*/', table_def, flags=re.DOTALL)
        
        # Build metadata
        metadata = {
            "table_name": table_name,
            "jsonb_fields": list(set(jsonb_fields)),
            "related_tables": list(set(relationships)),
            "table_comments": table_comments,
            "object_type": "complete_table_definition"
        }
        
        # Create single chunk per table with full definition
        chunks.append(Document(
            page_content=table_def.strip(),
            metadata=metadata
        ))
        seen_tables.add(table_name)
    
    return chunks

# Usage (replace with your actual content loading):
with open("db_schema.sql") as f:
    content = f.read()
    
final_chunks = process_tables(content)
#print(f"Total tables processed: {len(final_chunks)}")


def process_schema(db):
    """Process schema with enhanced comment extraction and relationship detection"""
    print("Starting schema processing...")
    ddls = []
    relationships = {}
    jsonb_key_variations = {}
    
    # Load and parse DDL file
    with open("db_schema.sql", "r", encoding="utf-8") as f:
        ddl_content = f.read()
    #print(f"Loaded schema file with {len(ddl_content)} characters")
    
    # Split on CREATE TABLE while keeping preceding comments
    table_defs = re.split(r'(?=\nCREATE TABLE)', ddl_content)
    #print(f"Found {len(table_defs)} potential table definitions")
    #print(f"Total chunks created: {len(table_defs)}")

    # First pass: collect all columns for common column analysis
    table_columns = {}
    for table_def in table_defs:
        if not table_def.strip() or "CREATE TABLE" not in table_def:
            continue
        table_name_match = re.search(r'CREATE TABLE public\.(dev_\w+)', table_def)
        if table_name_match:
            table_name = table_name_match.group(1)
            columns = []
            for line in table_def.split('\n'):
                col_match = re.match(r'^"?(\w+)"?', line.strip())
                if col_match and "CONSTRAINT" not in line:
                    columns.append(col_match.group(1))
            table_columns[table_name] = columns

    # Second pass: process each table definition
    for i, table_def in enumerate(table_defs):
        if not table_def.strip() or "CREATE TABLE" not in table_def:
            continue
            
        # Extract table name with schema
        table_name_match = re.search(r'CREATE TABLE public\.(dev_\w+)', table_def)
        if not table_name_match:
            continue
            
        table_name = table_name_match.group(1)
        #print(f"\nProcessing table: {table_name}")

        # Extract preamble comments
        preamble_comments = []
        table_description = "No description"

        # Process columns and comments
        col_comments = {}
        foreign_keys = []
        columns = []
        table_jsonb_fields = {}  # Now table-specific
        
        for line in table_def.split('\n'):
            line = line.strip()
            
            # Extract column with type and comment (Regex parsing)
            col_match = re.match(r'^"?(\w+)"?\s+([^\s-]+?)\s*[^--]*--\s*(.*)', line)
            if col_match:
                col_name, col_type, comment = col_match.groups()
                col_comments[col_name] = f"{comment.strip()} (type: {col_type})"
                columns.append(col_name)
                #print(f"Found column {col_name} ({col_type}) with comment: {comment}")
                
                # If column is JSONB, extract its fields
                if col_type.lower() == 'j':
                    #print(f"Processing JSONB column: {col_name}")
                    try:
                        # SQL query to get JSONB fields
                        query = f"""
                            SELECT key
                            FROM (SELECT {col_name} FROM {table_name} LIMIT 1) AS subquery, 
                            LATERAL jsonb_object_keys(subquery.{col_name}) AS key;
                        """
                        #print(f"Executing JSONB field extraction query: {query}")
                        result = db.run(query)
                        if result:
                            # Handle different result formats
                            if isinstance(result, str):
                                try:
                                    import json
                                    result = json.loads(result)
                                except json.JSONDecodeError:
                                    if result.startswith('[') and result.endswith(']'):
                                        result = [key.strip(" '\"") for key in result[1:-1].split(',')]
                                    else:
                                        result = [key.strip() for key in result.split(',')]
                            
                            if isinstance(result, list):
                                table_jsonb_fields[col_name] = []
                                for row in result:
                                    # Keep original nested structure by removing split on '.' and just cleaning
                                    clean_field = re.sub(r"^[^a-zA-Z0-9.]+|[^a-zA-Z0-9.]+$", "", str(row))
                                    if clean_field:
                                        table_jsonb_fields[col_name].append(clean_field)
                                table_jsonb_fields[col_name] = list(set(table_jsonb_fields[col_name]))
                            else:
                                table_jsonb_fields[col_name] = list(result.keys()) if isinstance(result, dict) else []
                            
                            #print(f"Found {len(table_jsonb_fields[col_name])} JSONB fields for {col_name}: {table_jsonb_fields[col_name]}")
                        else:
                            #print(f"No JSONB fields found for {col_name}")
                            table_jsonb_fields[col_name] = []
                    except Exception as e:
                        #print(f"Error extracting JSONB fields from {col_name}: {str(e)}")
                        table_jsonb_fields[col_name] = []
                    print(f"Final JSONB fields for {col_name}: {table_jsonb_fields[col_name]}")
            
            # Extract foreign keys
            fk_match = re.search(r'FOREIGN KEY\s*\([^)]+\)\s*REFERENCES\s*public\.(dev_\w+)', line)
            if fk_match:
                foreign_key = fk_match.group(1)
                foreign_keys.append(foreign_key)
                #print(f"Found foreign key to: {foreign_key}")

        # Find common column relationships (excluding id)
        common_columns_map = {}
        filtered_columns = set(col for col in columns if col.lower() != 'id')
        for other_table, other_columns in table_columns.items():
            if other_table == table_name:
                continue
            filtered_other = set(col for col in other_columns if col.lower() != 'id')
            common = filtered_columns.intersection(filtered_other)
            if common:
                common_columns_map[other_table] = list(common)
                #print(f"Common columns with {other_table}: {common}")

        # Build metadata
        all_relations = list(set(foreign_keys + list(common_columns_map.keys())))
        metadata = {
            "table_name": table_name,
            "related_tables": ", ".join(all_relations),
            "column_descriptions": str(col_comments),
            "common_columns": str(common_columns_map),
            "jsonb_fields": str(table_jsonb_fields)  # Now only contains current table's fields
        }
        #print(f"Generated metadata for {table_name} with {len(table_jsonb_fields)} JSONB columns")

        # Create formatted content
        content_parts = [
            f"/* Table: {table_name} */",
            f"/* This table contains {len(columns)} columns and is related to: {metadata['related_tables']} */",
            f"CREATE TABLE public.{table_name} (",
            *[f"  {col} -- {desc}" for col, desc in col_comments.items()],
            ");",
            f"/* Relationships: {metadata['related_tables']} */",
            f"/* Common Columns: {metadata['common_columns']} */",
            f"/* JSONB Fields: {metadata['jsonb_fields']} */"
        ]
        #print(f"Created content with {len(content_parts)} parts including JSONB fields")

        # Create document with filtered metadata
        chunk = Document(
            page_content="\n".join(content_parts),
            metadata={k: v for k, v in metadata.items() 
                     if isinstance(v, (str, int, float, bool))}
        )
        #print(f"Created document for {table_name} with JSONB metadata")
        
        ddls.append(chunk)
        relationships[table_name] = all_relations

    #print(f"\nProcessed {len(ddls)} tables with relationships:")
    for table, rels in relationships.items():
        print(f"  {table}: {rels}")
    
    #print(f"Total JSONB fields processed across all tables: {sum(len(eval(d.metadata.get('jsonb_fields', '{}'))) for d in ddls)}")
    return ddls, relationships, jsonb_key_variations


class SchemaRetriever:
    def __init__(self, vector_store, relationship_graph):
        #print("Initializing SchemaRetriever with vector store and relationship graph")
        self.vector_store = vector_store
        self.relationship_graph = relationship_graph
        self.base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )
        print("Base retriever configured with k=10")

    def find_related_tables(self, table_name):
        #print(f"\nFinding related tables for {table_name} with JSONB field analysis")
        collection = self.vector_store._collection.get()
        related = []
        
        #print(f"Scanning {len(collection['documents'])} documents for relationships and JSONB fields")
        for doc, metadata in zip(collection['documents'], collection['metadatas']):
            if metadata["table_name"] in self.relationship_graph.get(table_name, []):
                jsonb_fields = metadata.get("jsonb_fields", "None")
                #print(f"Found related table: {metadata['table_name']} with JSONB fields: {jsonb_fields}")
                related.append(Document(
                    page_content=doc,
                    metadata=metadata
                ))
        
        #print(f"Total related tables found: {len(related)} with JSONB field analysis complete")
        return related

    def get_relevant_documents(self, query):
        #processed_query = preprocess_query(query)
        #print(f"Processed query: {processed_query}")
        #print(f"\nGetting relevant documents for query: '{query}' with JSONB field analysis")
        
        # Get initial results with scores
        #print("Performing similarity search with scores and JSONB field analysis...")
        docs_scores = self.vector_store.similarity_search_with_score(query, k=10)
        #print(f"Initial search returned {len(docs_scores)} results with JSONB field metadata")
        
        # Adjust score threshold and include JSONB field relevance
        filtered_docs = []
        #print("Filtering documents with score < 0.7 and calculating JSONB field relevance...")
        for doc, score in docs_scores:
            if score < 0.7:
                # Calculate JSONB field relevance
                jsonb_relevance = 0
                jsonb_fields = doc.metadata.get("jsonb_fields", "")
                if isinstance(jsonb_fields, str):
                    jsonb_fields = jsonb_fields.lower()
                    # Check if query terms match JSONB field names
                    query_terms = query.lower().split()
                    jsonb_relevance = sum(0.3 for term in query_terms if term in jsonb_fields)
                    #print(f"JSONB field analysis for {doc.metadata['table_name']}:")
                    #print(f"  Query terms: {query_terms}")
                    #print(f"  JSONB fields: {jsonb_fields}")
                    #print(f"  Calculated relevance: {jsonb_relevance:.2f}")
                
                # Add JSONB relevance to metadata
                doc.metadata["jsonb_relevance"] = min(1.0, jsonb_relevance)
                filtered_docs.append(doc)
                #print(f"Added document {doc.metadata['table_name']} with score {score:.2f} and JSONB relevance {jsonb_relevance:.2f}")
        
        #print(f"After filtering, {len(filtered_docs)} documents remain with JSONB field analysis")
        
        expanded = []
        seen = set()
        #print("Expanding results with related tables and their JSONB fields...")
        
        for doc in filtered_docs:
            tbl = doc.metadata.get("table_name")
            if tbl and tbl not in seen:
                expanded.append(doc)
                seen.add(tbl)
                #print(f"Adding base table: {tbl} with JSONB fields: {doc.metadata.get('jsonb_fields', 'None')}")
                # Add related tables
                related = self.find_related_tables(tbl)
                expanded += [d for d in related if d.metadata["table_name"] not in seen]
                #print(f"Added {len(related)} related tables for {tbl} with JSONB field analysis")
        
        # Remove duplicates based on table name
        unique_results = []
        seen_tables = set()
        #print("Removing duplicate tables while preserving JSONB field metadata...")
        for doc in expanded:
            tbl = doc.metadata["table_name"]
            if tbl not in seen_tables:
                unique_results.append(doc)
                seen_tables.add(tbl)
        
        #print(f"After deduplication, {len(unique_results)} unique tables remain with JSONB field metadata")
        
        # Sort by importance including JSONB relevance
        #print("Sorting results by importance with JSONB field relevance as primary factor...")
        sorted_results = sorted(unique_results, 
                              key=lambda x: (
                                  -x.metadata.get("jsonb_relevance", 0),
                                  -int(x.metadata.get("is_central_table", 0)),
                                  -int(x.metadata.get("column_count", 0)),
                                  -int(x.metadata.get("is_fact_table", 0))
                              ))[:7]
            
        return sorted_results

class EnhancedSchemaRetriever(SchemaRetriever):
    def get_relevant_documents(self, query: str):
        # Get base results
        docs = super().get_relevant_documents(query)
        
        # Enhance with comment and JSONB scoring
        for doc in docs:
            # Initialize score if not present
            doc.metadata.setdefault("retrieval_score", 0.0)
            
            # Score based on column comments
            comment_matches = sum(
                1 for comment in doc.metadata.get("column_comments", {}).values()
                if query.lower() in comment.lower()
            )
            
            # Handle JSONB field format variations
            jsonb_matches = 0
            jsonb_fields = doc.metadata.get("jsonb_fields", {})
            
            # Convert string representation to dict if needed
            if isinstance(jsonb_fields, str):
                try:
                    jsonb_fields = ast.literal_eval(jsonb_fields)
                except (ValueError, SyntaxError):
                    jsonb_fields = {}

            # Process JSONB fields
            if isinstance(jsonb_fields, dict):
                for col, keys in jsonb_fields.items():
                    if isinstance(keys, (list, tuple)):
                        jsonb_matches += sum(1 for key in keys if query.lower() in key.lower())
                    elif isinstance(keys, str):
                        jsonb_matches += sum(1 for key in keys.split(', ') if query.lower() in key.lower())
            
            # Boost score for relevant metadata
            doc.metadata["retrieval_score"] += (comment_matches * 0.2) + (jsonb_matches * 0.3)
        
        # Re-sort documents with enhanced scoring
        return sorted(docs, key=lambda x: -x.metadata["retrieval_score"])

def format_chunk_with_metadata(doc: Document) -> str:
    """Enhance chunk formatting with embedded metadata"""
    content = doc.page_content
    metadata = doc.metadata
    
    # Add column comments
    if "column_comments" in metadata:
        comments = "\n".join(
            f"Column {col}: {comment}"
            for col, comment in metadata["column_comments"].items()
        )
        content += f"\n\nColumn Descriptions:\n{comments}"
    
    # Add JSONB fields with type handling
    if "jsonb_fields" in metadata:
        jsonb_fields = metadata["jsonb_fields"]
        
        # Convert string representation to dict if needed
        if isinstance(jsonb_fields, str):
            try:
                jsonb_fields = ast.literal_eval(jsonb_fields)
            except (ValueError, SyntaxError):
                jsonb_fields = {}

        # Format JSONB information
        if isinstance(jsonb_fields, dict):
            jsonb_info = []
            for col, keys in jsonb_fields.items():
                if isinstance(keys, (list, tuple)):
                    jsonb_info.append(f"JSONB {col}: {', '.join(keys)}")
                elif isinstance(keys, str):
                    jsonb_info.append(f"JSONB {col}: {keys}")
            
            if jsonb_info:
                content += f"\n\nJSONB Fields:\n" + "\n".join(jsonb_info)
    
    return content



loader = TextLoader("db_schema.sql", encoding="utf-8")
documents = loader.load()

splitter = SQLTextSplitter()
chunks = splitter.split_documents(documents)

#embedding_model = CohereEmbeddings(model="embed-english-v3.0")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
chunks, rel_graph, jsonb_keys = process_schema(db)  

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="openai_schema",
    persist_directory="./chroma_openai",
    collection_metadata={"hnsw:space": "cosine"}
)

# After creating the vector_store, add this to print stored chunks
#print("\nStored chunks in Chroma:")
#collection = vector_store._collection.get()
#chunk_count = len(collection['documents'])  # Get total chunk count
#print(f"Total chunks stored: {chunk_count}\n")

#for i, (doc, metadata) in enumerate(zip(collection['documents'], collection['metadatas']), 1):
    #print(f"Chunk {i}/{chunk_count}:")
    #print(f"Document content:\n{doc}")
    #print(f"Metadata:\n{metadata}")
    #print("="*50)

# Add this cell after your vector_store initialization
#collection = vector_store._collection.get()
#print(f"\nTotal chunks in ChromaDB: {len(collection['documents'])}\n")  


def create_metadata_aware_sql_chain(llm, db, retriever):
    template = """Given an input question, first analyze the schema metadata comments, then generate a SQL query using ONLY these steps:
    1. Identify relevant tables using their COMMENT metadata
    2. Check column COMMENTS to understand relationships
    3. Use ONLY columns explicitly defined in tables
    4. Reference JSONB fields using -> operator
    5. Use this schema:
    {schema}
    
    
    Guidelines:
    1. Use explicit JOIN syntax with proper ON conditions based on vector relationships
    2. Include relevant WHERE clauses using embedded column descriptions
    3. Use table aliases for clarity
    4. Prioritize filtering using indexed columns (tenant_id, case_id, order_id,location_id,account_id)
    5. Return ONLY the SQL query without any explanations
    6. Handle date comparisons using ISO format (YYYY-MM-DD)
    7. Use column descriptions from embeddings to determine appropriate filters
    8. Consider table relationships from vector similarity when joining
    9. Handle case variations using LOWER(): LOWER(metadata->>'key') = 'lowercase value'
    10. Account for nested JSON paths: metadata->'nested'->>'field'
    11. Use ILIKE for partial matches: metadata->>'key' ILIKE '%pattern%'
    12. Consider common value transformations:
      * Trim whitespace
      * Remove special characters
      * Expand abbreviations (thirds → 3rds) // Add this line
    13. ALWAYS wrap JSONB access in parentheses before casting:
      (metadata->>'field')::TIMESTAMPTZ

       CRITICAL DATE HANDLING RULES:
    1. ALWAYS cast JSONB date strings: (metadata->>'date_field')::TIMESTAMPTZ
    2. Compare dates using ISO-8601 format: YYYY-MM-DDTHH:MI:SSZ
    3. Use AT TIME ZONE for conversions: 
       (metadata->>'date_field')::TIMESTAMPTZ AT TIME ZONE 'UTC'

    MAKE SURE TO ONLY USE COLUMNS FROM THE GIVEN TABLES ONLY AND DO NOT GUESS BY YOURSELF.
    USE THE EMBEDDED COLUMN DESCRIPTIONS TO DETERMINE APPROPRIATE DATA TYPES AND FILTERS.

    DO NOT include any markdown formatting, code blocks, or explanation text such as ```sql in start and ``` in end of the SQL query.
    ONLY return the SQL query itself, nothing else. The SQL query should be able to run directly on postgresql
    The query should be executable and return the results asked in the question.
    
    {context}

    Guidelines:
    1. Use columns with matching descriptions first
    2. Prefer JSONB fields from the listed options
    3. Match query terms to column comments
    4. Use this exact format for JSONB: metadata->>'exactKey'

    Question: {question}
    SQL Query:"""
    
    prompt = PromptTemplate.from_template(template)
    
    return (
        RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(
                format_chunk_with_metadata(d) 
                for d in retriever.get_relevant_documents(x["question"])
            )
        )
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

example_metadata = """
-- dev_tenants table --
- tenant_id: varchar (references practices)
- practice_name: varchar

-- dev_cases table -- 
- tenant_id: varchar (foreign key to dev_tenants)
- data: jsonb (contains case details)
"""

retriever = EnhancedSchemaRetriever(vector_store, rel_graph)


#streamlit framework

st.title('Text to SQL Demo With Cohere API')
llm_query=st.text_input("Enter the query u want to answer about:")
chain = create_metadata_aware_sql_chain(llm, db, retriever)
result = chain.invoke({
    "question": f"{llm_query} - ALWAYS use tenant_id for practice references",
    "schema": example_metadata  # Pass parsed metadata
})

if llm_query:
    import re
    # Clean the result using regex
    cleaned_result = re.sub(r'^```sql\s*|\s*```$', '', result).strip()
    
    st.markdown("**Generated SQL Query:**")
    st.code(cleaned_result, language='sql')
    st.markdown("**Query Results:**")
    st.write(db.run(cleaned_result))

    