## Machine Learning with Embeddings
You can use embeddings to
- Evaluate text similarity, see [test_sentence_similarity.py](test_sentence_similarity.py)
- Build your own classifier, see [test_classification.py](test_classification.py)
- Search relative texts, see [test_semantic_search.py](test_semantic_search.py)

To these tests, you need to download the data [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). You also need an OpenAI API key for comparison.

Run with:
```bash
cd playground/test_embedding
python3 test_classification.py
```

The script will train classifiers based on `vicuna-7b`, `text-similarity-ada-001` and `text-embedding-ada-002` and report the accuracy of each classifier.
