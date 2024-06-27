from pdf_helper import load_pdf_data, split_docs

#based on https://ollama.com/blog/embedding-models

print('Testing OLLAMA embeddings')

import ollama
import chromadb

doc = load_pdf_data('./files/comments.pdf')
documents = split_docs(doc)
documents_cnt = len(documents)
print(f'Number of documents: {documents_cnt}')


client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  document_text = d.page_content
  response = ollama.embeddings(model="nomic-embed-text", prompt=document_text)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[document_text]
  )

print('embeddings generated')
# an example prompt
#prompt = "What animals are llamas related to?"
prompt = "Should I write comments in code?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

output = ollama.generate(
  model="mistral",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
