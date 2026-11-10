import os
import chromadb

from sentence_transformers import SentenceTransformer

# Step 1: Configuration
DATABASE_FOLDER = "chroma_db"
DOCUMENT_FOLDER = "dataset/documents"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Step 2: Load embedding model
print("Loading Sentence Transformer Model...")

model = SentenceTransformer(MODEL_NAME)

print("Model Loaded Successfully!")

# Step 3: Connect ChromaDB
client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

print("ChromaDB Connected Successfully!")

# Step 4: Create or access collection
collection = client.get_or_create_collection(
    name="semantic_documents"
)

print("Collection Ready!")

# Step 5: Load documents
documents = []
ids = []

for filename in os.listdir(DOCUMENT_FOLDER):

    if filename.endswith(".txt"):

        filepath = os.path.join(
            DOCUMENT_FOLDER, filename
        )

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(content)
        ids.append(filename)

print("Total Documents Loaded:", len(documents))

# Step 6: Generate embeddings
print("\nGenerating Document Embeddings...")

embeddings = model.encode(documents).tolist()

print("Embeddings Generated Successfully!")

# Step 7: Store documents in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents Stored in ChromaDB!")

print("Total Documents in Database:", collection.count())
# STEP 8: Perform Semantic Search

query = input("\nEnter your search query: ")

# Convert query into embedding
query_embedding = model.encode([query]).tolist()

# Retrieve Top-5 documents
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

print("\n===== SEMANTIC SEARCH RESULTS =====")

for i, doc in enumerate(results["documents"][0]):

    print(f"\nResult {i+1}")
    print("Document:", results["ids"][0][i])
    print("Content:", doc[:200])

    distance = results["distances"][0][i]
    print("Distance Score:", distance)
    # STEP 9: Keyword-Based Search

print("\n===== KEYWORD-BASED SEARCH =====")

query_words = query.lower().split()

stop_words = {
    "what", "is", "the", "how", "do",
    "can", "a", "an", "are", "from",
    "to", "and", "of", "in"
}

keywords = [
    word for word in query_words
    if word not in stop_words
]

keyword_results = []

for i, doc in enumerate(documents):

    content = doc.lower()

    score = sum(
        content.count(word)
        for word in keywords
    )

    keyword_results.append(
        (ids[i], score, doc)
    )

keyword_results.sort(
    key=lambda x: x[1],
    reverse=True
)

for i, result in enumerate(keyword_results[:5]):

    print(f"\nResult {i+1}")
    print("Document:", result[0])
    print("Keyword Score:", result[1])
    print("Content:", result[2][:200])
    import pandas as pd
    # STEP 10: Save All Search Results in CSV

print("\n===== PROCESSING ALL 15 QUERIES =====")

# Load user queries
query_data = pd.read_csv("dataset/queries.csv")

search_results = []
similarity_scores = []

stop_words = {
    "what", "is", "the", "how", "do",
    "can", "a", "an", "are", "from",
    "to", "and", "of", "in"
}

for _, row in query_data.iterrows():

    query_id = row["query_id"]
    user_query = row["query"]

    print("\nProcessing:", query_id, user_query)

    # Generate query embedding
    query_embedding = model.encode(
        [user_query]
    ).tolist()

    # Semantic Search - Top 5
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    for rank, doc_id in enumerate(results["ids"][0]):

        distance = results["distances"][0][rank]

        # Transformed distance-based relevance score
        similarity = 1 / (1 + distance)

        search_results.append({
            "query_id": query_id,
            "query": user_query,
            "search_type": "Semantic",
            "rank": rank + 1,
            "document_id": doc_id
        })

        similarity_scores.append({
            "query_id": query_id,
            "query": user_query,
            "document_id": doc_id,
            "distance_score": distance,
            "similarity_score": similarity
        })

    # Keyword Search
    keywords = [
        word for word in user_query.lower().split()
        if word not in stop_words
    ]

    keyword_results = []

    for doc_id, doc in zip(ids, documents):

        score = sum(
            doc.lower().count(word)
            for word in keywords
        )

        keyword_results.append((doc_id, score))

    keyword_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (doc_id, score) in enumerate(
        keyword_results[:5]
    ):

        search_results.append({
            "query_id": query_id,
            "query": user_query,
            "search_type": "Keyword",
            "rank": rank + 1,
            "document_id": doc_id
        })

# Save search results
results_df = pd.DataFrame(search_results)

results_df.to_csv(
    "dataset/search_results.csv",
    index=False
)

# Save similarity scores
scores_df = pd.DataFrame(similarity_scores)

scores_df.to_csv(
    "dataset/similarity_scores.csv",
    index=False
)

print("\n===================================")
print("ALL SEARCH RESULTS SAVED!")
print("===================================")

print("Total Queries:", len(query_data))
print("Total Search Results:", len(results_df))
print("Total Similarity Records:", len(scores_df))

print("\nCSV Files Created Successfully!")