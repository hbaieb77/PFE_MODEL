import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('tokenizer')
    model = BertForSequenceClassification.from_pretrained("hbaieb77/nlpmodel")
    return tokenizer,model

tokenizer,model = get_model()


def predict(text, model, tokenizer, max_len):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=False,
        truncation=True
    )
    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return label_dict[predicted_class]

label_dict = {0: 'Analytics', 1: 'COM BCP BASE CLIENT PARTENAIRE'}

MAX_LEN = 128


user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")



if user_input and button :
    predicted_group = predict(user_input, model, tokenizer, MAX_LEN)
    st.write("Prediction: ",predicted_group)