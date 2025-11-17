# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# model_name = "google/flan-t5-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Load a SentenceTransformer model for embedding
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Example corpus
# documents = [
#     "Python is a popular programming language.",
#     "T5 stands for Text-to-Text Transfer Transformer.",
#     "RAG is a framework for question answering."
# ]

# # Generate embeddings
# embeddings = embedding_model.encode(documents, convert_to_tensor=False)

# # Build FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)  # L2 similarity
# index.add(np.array(embeddings))

# def retrieve(query, k=3):
#     query_embedding = embedding_model.encode([query], convert_to_tensor=False)
#     distances, indices = index.search(np.array(query_embedding), k)
#     return [documents[i] for i in indices[0]]

# # Test the retriever
# query = "What is T5?"
# retrieved_docs = retrieve(query)
# print(retrieved_docs)

# def generate_answer(query, retrieved_docs):
#     # Prepare input by concatenating query with retrieved context
#     context = " ".join(retrieved_docs)
#     input_text = f"Question: {query} Context: {context}"

#     # Tokenize and generate answer
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Generate an answer
# answer = generate_answer(query, retrieved_docs)
# print("Answer:", answer)

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize ChromaDB client
client = chromadb.Client()

# Create or connect to a collection (index) in ChromaDB
collection_name = "documents_collection"
collection = client.create_collection(collection_name)

# Initialize Sentence-Transformer for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Flan-T5 Large model
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example corpus of documents
documents = [
    "What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."
]

# Generate embeddings for the documents
embeddings = embedding_model.encode(documents, convert_to_tensor=False)

# Create unique IDs for each document (e.g., using a simple index or hash)
# ids = [str(i) for i in range(len(documents))]  # Simple numeric string IDs

# Insert documents and embeddings into ChromaDB
collection.add(
    ids=[str(i) for i in range(len(documents))],  # List of unique IDs
    documents=documents,  # List of document texts
    metadatas=[{"content": doc} for doc in documents],  # List of metadata dictionaries
    embeddings=embeddings  # List of embeddings
)

# Retrieve documents based on query
def retrieve(query, k=3):
    # Encode the query to get its embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)

    # Perform a similarity search in the ChromaDB collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    # Flatten the nested list of documents
    retrieved_docs = [doc for docs in results['documents'] for doc in docs]
    return retrieved_docs

# Generate an answer using Flan-T5 Large
def generate_answer(query, retrieved_docs):
    # Format context with explicit structure
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])

    # Prepare the prompt
    prompt = f"""
    You are a helpful assistant. Answer the question strictly based on the context provided below. If the answer cannot be found in the context, respond with I don't have knowledge about that topic.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Tokenize and generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=2, early_stopping=True)
    final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_answer


# Example query
query = "What is the past medical history of the patient?"
# query = "What is the name of the patient?"
retrieved_docs = retrieve(query)
# print("Retrieved Documents:", retrieved_docs)
print("Question:", query)

# Generate an answer
answer = generate_answer(query, retrieved_docs)
print("Answer:", answer)
