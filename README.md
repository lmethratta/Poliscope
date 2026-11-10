# Poliscope

**A politically balanced search engine: information retrieval with bias control.**

Poliscope ranks political statements by both how relevant they are to a search query and which way they lean, left or right. The goal is to help users find viewpoints that match or challenge their own, and to make it easier to compare both sides of an issue instead of staying in an echo chamber.

Team project for Northeastern University's information retrieval course (Spring 2025).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmethratta/Poliscope/blob/main/model.ipynb)

## How it works

1. **Bias classifier:** text is vectorized with TF-IDF (lowercased, punctuation and stopwords removed), and a logistic regression model, tuned with GridSearchCV, predicts whether it leans left or right. Its predicted probability is converted to a bias score from −1 (left) to +1 (right).
2. **Relevance scoring:** BM25 (Okapi) scores how well each statement matches the query.
3. **Score fusion:** the two scores are combined into one ranking, using one of three strategies:
   - Multiplicative: BM25 × bias
   - Additive: BM25 + bias
   - Weighted: 0.6 × BM25 + 0.4 × bias
The ranking pipeline uses the logistic regression classifier for its bias scores: it is fast, accurate and easy to interpret.

**Transformer experiment:** separately from the pipeline, the notebook also fine-tunes a DistilBERT model (Hugging Face Transformers, PyTorch) as a regression model that outputs a bias score directly, and tests it on example sentences.

## Dataset

[TwinViews-13k](https://huggingface.co/datasets/twinviews/twinviews-13k): 13,855 pairs of left- and right-leaning statements on the same topic, across 1,000+ topics such as healthcare, taxation and climate change. Each pair is split into two labeled statements (about 27,700 in total), with 90% used for training and 10% for validation.

## Results

**Bias classifier (logistic regression):** 97% accuracy, precision and recall on the validation set, with balanced errors across both classes. The notebook includes the confusion matrix, ROC and precision–recall curves, and the words most associated with each side.

**Ranking:** evaluated with MAP@10 and NDCG@10 on three queries: "healthcare for all citizens", "tax cuts for economic growth" and "environmental policy and climate change". A statement counts as relevant when its dataset topic matches the query's subject.

| Fusion strategy | MAP@10 | NDCG@10 |
|---|---|---|
| Multiplicative | 0.77 | 0.61 |
| Additive | 0.70 | 0.71 |
| Weighted (0.6 / 0.4) | 0.70 | 0.69 |

Multiplicative ranking puts the most relevant results near the top (higher MAP), while additive ranking orders them best overall (higher NDCG).

## Run it

Open the notebook in Google Colab with the badge above, set **Runtime → Change runtime type → T4 GPU**, and select **Runtime → Run all**. A GPU is needed for the DistilBERT step. The notebook installs its own dependencies (`datasets`, `transformers`, `rank_bm25`).

## Limitations

- Bias is binary (left or right); there is no neutral or centrist class, which can flatten nuanced viewpoints.
- The dataset is short opinion statements, so performance on full news articles would likely be lower.
- The DistilBERT model learned the dataset's full-sentence style and can misjudge short phrases that differ from it.
- Ranking was evaluated on three queries, with relevance based on topic labels rather than human judgments.

## Future work

- A continuous or multi-class bias scale that includes neutral content
- Evaluation with human relevance judgments and more queries
- Learned ranking models and tuning of the fusion weights

## Team

- **Leah Methratta:** TF-IDF + logistic regression bias classifier (training, validation, precision/recall, confusion matrix); MAP@10 and NDCG@10 evaluation
- **Dilan Kudva:** BM25 indexing and query processing; MAP@10 and NDCG@10 evaluation
- **Eruba Ekedayen:** score fusion strategies, grid-search hyperparameter tuning, performance and error analysis
