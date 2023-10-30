# Knowledge Extraction
This package can be used to extract relevant subgraphs from a knowledge graph to a given string query.

## Installation
pip install -r requirements.txt

### Usage example
In order to run the package and knowledge_extractor.py with the dataset loaded in the example file an openai api key must be provided. This is use-case specific, as the nodes in the example's kg have been embedded with the openai embeddings model.

create .env file and make sure in contains a field with your OpenAI key in the format: 

OPENAI_API_KEY=OPENAI_API_KEY=sk-xxxxxxxxx

Work In Progress