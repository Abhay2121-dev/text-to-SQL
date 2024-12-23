import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/content/drive/MyDrive/l3_finagent/l3_finagent_step60",  # Update this path
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# Define prompt template
ft_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is a user question, paired with retrieved context. Write a response that appropriately answers the question,
include specific details in your response. <|eot_id|>

<|start_header_id|>user<|end_header_id|>

### Question:
{}

### Context:
{}

<|eot_id|>

### Response: <|start_header_id|>assistant<|end_header_id|>
{}"""

# Inference function
def inference(question, context):
    inputs = tokenizer(
        [
            ft_prompt.format(
                question,
                context,
                "",  # Output placeholder
            )
        ], return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.batch_decode(outputs)
    return response

# Extract the response text
def extract_response(text):
    text = text[0]
    start_token = "### Response: <|start_header_id|>assistant<|end_header_id|>"
    end_token = "<|eot_id|>"

    start_index = text.find(start_token) + len(start_token)
    end_index = text.find(end_token, start_index)

    if start_index == -1 or end_index == -1:
        return None

    return text[start_index:end_index].strip()

# Streamlit UI
st.title("Text-to-SQL Query Generator with Fine-tuned LLaMA 3 Model")

context = st.text_area("Enter Database Schema (Context):", placeholder="CREATE TABLE ...")
question = st.text_area("Enter Your SQL Query Question:", placeholder="What is the total value ...")

if st.button("Generate SQL Query"):
    if context and question:
        with st.spinner("Generating Response..."):
            resp = inference(question, context)
            parsed_response = extract_response(resp)
            st.text_area("Generated SQL Query:", value=parsed_response, height=200)
    else:
        st.warning("Please enter both context and question to generate a query.")
