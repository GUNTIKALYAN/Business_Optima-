# Business Optima Assignment

### Overview 
- From all the given task options, I have chosen the Option 1 : Smart Text Classifier
- This assignment categorizes customer support message into predifined business-related categories using two approaches:
- **Traditional ML**
- **Using Pretrained Model (DistilBERT)**

### Dataset

``` https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset/data ```

**Dataset preparation:**
- A subset of **5 intent categories** was selected from the target column.
- Only relevant columns required for intent prediction were retained.
- Columns were renamed for consistency:
  - `utterance → text`
  - `intent → label`
 
**Selected intent classes:**
- `change_order`
- `change_shipping_address`
- `check_refund_policy`
- `contact_human_agent`
- `delivery_period`

### Approach :

### Traditional ML Pipeline
The solution demonstrate an end-to-end NLP pipeline including:
- Data preprocessing and exploration
- Text Vectorization using TF-IDF
- Model training and evaluation
- Deployment via a FastAPI

### Pretrained :
- Model used: distilbert-base-uncased
- Fine-tuned for sequence classification on the same dataset
- Trained entirely on CPU using a lightweight configuration
- Label mappings handled explicitly for inference

### Evaluation Metrics :

The following metrics were used to evaluate model performance in ``` Model.ipynb```
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

# Observations
- Both the TF-IDF model and the fine-tuned DistilBERT model achieved similar performance (Accuracy ≈ 1.0) on the test set.
- Because the dataset is relatively small, clean, and intent-specific, making it easy for both lexical and contextual models to separate classes.

### API Implementation

Framework 
- FastAPI
- Pydantic for request/response validation

### Endpoint 

``` POST /predict ``` -> for the Tranditional ML 

### Request Body

![Request](./images/request.png)

### Response Body

![Response](./images/response.png)



### How to Run 
### For the Tradition ML
- Install Dependencies
  ``` pip install -r requirements.txt ```

- Start the API
  ``` uvicorn app:app --reload ```

- Open Swagger UI
  ``` http://127.0.0.1.8000/docs ```

  ### For the Pretrained Model
- Install Dependencies
  ``` pip install -r requirements.txt ```

- Go to pretrained folder
  ``` cd pretrained ```

- Start the API
  ``` uvicorn app:app --reload ```

- Open Swagger UI
  ``` http://127.0.0.1.8000/docs ```
  

### Example Sentences
- I want to add another item to my existing order

- How long will it take for my order to arrive?

- Under what conditions can I get a refund?

- I want to speak with a customer support agent

### Limitations 
- The dataset category values are about 40 for each category that may not fully represent real-world customer language
- TF-IDF is a lexical model and does not capture the semantic meaning

### Future Improvements
- Need to use contextual embeddings (ex: sentence-transfromers)
- Expand dataset with real-world customer quries
  
