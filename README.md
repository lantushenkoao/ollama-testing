This test project is built based on official Ollama documentation
https://ollama.com/blog/embedding-models

For simplicity there is no config and therefore Ollama should be 

1. Running on http://localhost:11434/
2. Have mistral and nomic-embed-text models installed

File to parse is located in ./files/Comments.pdf . The file is converted to text and then split into chunks. 
These chunks are then converted to embeddings.  