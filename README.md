Based on the extracted notebook, here's a **highly professional and detailed `README.md`** for your project:

---

#  Semantic FAQ Chatbot with Next Question Prediction

A smart, transformer-based chatbot that not only returns **the most relevant answer** to a user query but also predicts the **next likely question**, simulating a more human-like, contextual conversation. Built using `SentenceTransformers` and fine-tuned on a domain-specific FAQ dataset.

---

##  Project Highlights

*  **Semantic Search**: Matches user queries with paraphrased FAQ answers.
*  **Next Question Prediction**: Suggests the next most probable question based on context.
*  **Fine-Tuned**: Built on top of `all-MiniLM-L6-v2` with domain-specific data.
*  **Optimized Training**: Uses `MultipleNegativesRankingLoss` for effective sentence embedding learning.

---

##  Dataset Overview

* **File**: `WITDS_FAQ_Paraphrased.xlsx`
* **Columns**:

  * `Question`: Paraphrased FAQ question
  * `Answer`: Corresponding answer
  * `Next Question`: Suggested follow-up question

The dataset is used for both **fine-tuning embeddings** and constructing a **next-question mapping**.

---

##  Setup & Installation

Install required libraries:

```bash
pip install sentence-transformers pandas torch openpyxl
```

---

##  File Structure

```
├── WITDS_FAQ_Paraphrased.xlsx   # Dataset file
├── semantic_faq_chatbot.ipynb   # Main training and inference notebook
├── output/                      # Fine-tuned model directory
├── README.md                    # Project documentation
```

---

##  Model Architecture

| Component           | Description                                  |
| ------------------- | -------------------------------------------- |
| Base Model          | `all-MiniLM-L6-v2` from SentenceTransformers |
| Loss Function       | `MultipleNegativesRankingLoss`               |
| Embedding Dimension | 384                                          |
| Training Epochs     | 5                                            |
| Batch Size          | 16                                           |

---

##  Training Workflow

```python
# Step 1: Load FAQ Data
faq_df = pd.read_excel("WITDS_FAQ_Paraphrased.xlsx")
faq_df = faq_df.dropna(subset=["Question", "Answer"]).fillna("")

# Step 2: Create Training Pairs
train_examples = [InputExample(texts=[q, q]) for q in faq_df["Question"]]

# Step 3: Fine-tune Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(DataLoader(train_examples, shuffle=True, batch_size=16), train_loss)],
    epochs=5,
    warmup_steps=10,
    show_progress_bar=True
)
```

---

##  Semantic Inference

After training, the model can perform semantic matching:

```python
from sentence_transformers import util

query = "How to apply for an internship?"
corpus = ["Visit the career portal", "Check our LinkedIn", "Email your resume"]

query_embedding = model.encode(query, convert_to_tensor=True)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
print("Best Match:", corpus[hits[0][0]['corpus_id']])
```

---

##  Next Question Prediction

Each question is mapped to a likely **next question**, forming a guided conversational path:

```python
next_question_map = dict(zip(faq_df["Question"], faq_df["Next Question"]))
next_q = next_question_map.get(query, "Is there anything else I can help with?")
```

This enables a chatbot to ask follow-ups like a human support agent.

---

##  Deployment (Suggestions)

* **Web UI**: Streamlit or Flask frontend
* **API**: FastAPI backend for semantic inference
* **Vector Indexing**: Use FAISS or SentenceTransformers `util.semantic_search` for larger datasets

---

##  Result

The final chatbot:

* Understands **semantic variations** of user queries.
* Retrieves the **most accurate answer**.
* Suggests a **next logical question** for seamless interaction.

---

##  Future Enhancements

*  Integrate with GPT-style models for next-question generation
*  Improve next-question mapping using a classifier or T5
*  Extend dataset with multilingual support
*  Add vector-based similarity search using FAISS for scale

---

## 👤 Author

**Bibhav Kumar**
Project built as part of Internship at **WITDS**
GitHub: [Bibhavcodeverse](https://github.com/Bibhavcodeverse)

---




