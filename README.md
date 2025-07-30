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



##  Key Features of the Chatbot

This chatbot goes far beyond basic query-answer matching. It replicates a **real-time smart assistant** with conversational context, memory, error tolerance, and user feedback collection.

---

###  1. **Semantic Answer Retrieval**

* The chatbot uses **`SentenceTransformer` embeddings** to find the **most semantically similar question** from a pre-trained FAQ dataset.
* Even if users **rephrase the question**, the chatbot can identify the underlying intent and provide the correct answer.
* It computes **cosine similarity** between user queries and existing FAQs to select the best answer.

```python
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
sims = util.cos_sim(query_vec, faq_embeddings)
best_match = torch.argmax(sims)
```

---

###  2. **Next-Question Prediction**

* After answering a question, the bot can optionally suggest the **next logical question**.
* This mimics a guided support conversation, improving UX and simulating **multi-turn dialogue**.

```python
next_question_map = dict(zip(faq_df["Question"], faq_df["Next Question"]))
```

**Example**:

> **Q**: How can I apply for an internship?
> **Bot**: You can apply through the official careers page.
> Would you like to know: *What documents are needed to apply?*

---

###  3. **Duplicate Question Detection**

* The chatbot keeps track of previously asked questions.
* If the user repeats a question (or a very similar one), the bot detects it and confirms if the previous answer was helpful.

```python
def is_duplicate(query):
    for prev_q in previous_questions:
        sim = util.cos_sim(query_vec, prev_vec).item()
        if sim > threshold:
            return True
```

**Bot Response**:

> You've already asked this or a similar question.
> Did that answer resolve your query? (yes/no)

---

###  4. **Irrelevant Question Handling**

* If the semantic similarity score is below a threshold (e.g., 0.5), the chatbot **flags the query as irrelevant**.
* It allows a limited number of irrelevant questions before politely ending the session.

```python
if best_score < IRRELEVANT_THRESHOLD:
    irrelevant_count += 1
    if irrelevant_count > MAX_IRRELEVANT_LIMIT:
        end_session()
```

---

###  5. **Dynamic Related Question Suggestions**

* If there's no next-question mapping, the bot suggests **top-3 semantically related FAQs** to continue the conversation.

```python
def get_related_questions(query, top_k=3):
    # Suggest questions closest in meaning to the user’s query
```

---

###  6. **User Contact Detail Capture**

* On successful resolution or exit, the bot asks for user **name**, **email**, and **phone number** (optional).
* It saves the data to `user_contacts.xlsx` for future follow-up or analysis.

```python
save_contact_to_excel(name, email, phone)
```

---

###  7. **Feedback Collection**

* At the end of the session or after timeout, users are asked to share their feedback.
* Feedback is stored in `user_feedback.xlsx` for further improvement of the chatbot.

```python
save_feedback_to_excel(feedback)
```

---

###  8. **Timeout Management**

* If the user stays inactive beyond a timeout period (default: 300 seconds), the session ends automatically.
* Feedback and contact collection are attempted before exit.

```python
inputimeout(prompt="You: ", timeout=TIMEOUT_SECONDS)
```

---

###  9. **Logging & Session Safety**

* Each interaction is **stateless per session**, but tracks questions asked to avoid duplication.
* Bot ensures clean exit via `SystemExit` after proper contact and feedback collection.

---

##  Example Interaction

```
 Chatbot loaded. Ask a question or type 'exit' to quit.
You: How do I apply for a job?
Bot: You can apply at our careers page.
Would you like to know: What documents are required?

You: What documents are required?
Bot: Please upload your resume, academic transcripts, and a cover letter.

You: exit
 Before you go, we'd love your feedback!
 Your Feedback: Great experience!
 Please share your contact details before exiting.
 Your Name: John Doe
 Your Email: john@example.com
 Your Phone (optional): 9876543210
 Bot: Thank you! Your details have been saved.
 Bot: Session ended after successful interaction.
```

---

##  Summary of Functional Flow

| Feature                  | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| Semantic Retrieval       | Understand and match user intent using embeddings  |
| Next Question Prediction | Suggest next question to guide the user            |
| Duplicate Detection      | Avoid repetitive interactions                      |
| Irrelevant Handling      | Control for off-topic questions                    |
| Feedback Capture         | Gather user experience insights                    |
| Contact Info Logging     | Optional user detail collection for future support |
| Timeout Detection        | Gracefully exit idle sessions                      |
| Related Questions        | Suggest meaningful follow-up questions             |

---



## 👤 Author

**Bibhav Kumar**
Project built as part of Internship at **WITDS**
GitHub: [Bibhavcodeverse](https://github.com/Bibhavcodeverse)

---

## 📜 Certificate

![Chatbot Project Certificate](certificate.png)

<img src="certificate.png" alt="Chatbot Project Certificate" width="600"/>




