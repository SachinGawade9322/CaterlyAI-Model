from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import json
import os
from pydantic import BaseModel
from PDF_Data_Extraction import TransactionProcessor  
from AIBankAssistance import extract_balance, extract_transactions, generate_dynamic_context, generate_response, load_finetuned_model

app = FastAPI(title="AI Banking Assistant API")

class UserQuery(BaseModel):
    query: str

# file form user
@app.post("/upload/")
async def upload_statement(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        processor = TransactionProcessor()
        json_data = processor.process_file(temp_path)

        if not json_data or "transactions" not in json_data:
            raise HTTPException(status_code=400, detail="No transactions found in the uploaded file.")

        output_path = "json_output/extracted_transactions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=4)

        os.remove(temp_path)  

        return {"message": "File processed successfully", "transactions": json_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Response 
@app.post("/analyze/")
async def analyze_finances(user_query: UserQuery):
    try:
        json_path = "json_output/extracted_transactions.json"
        if not os.path.exists(json_path):
            raise HTTPException(status_code=400, detail="No transactions found. Please upload a statement first.")

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            
        model, tokenizer = load_finetuned_model()

        balance_info = extract_balance(json_data)
        transactions = extract_transactions(json_data)
        context = generate_dynamic_context(json_data, user_query.query)

        # Generate response 
        response = generate_response(model, tokenizer, user_query.query, context, balance_info, transactions)

        return {"query": user_query.query, "response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing finances: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



# http://127.0.0.1:8000/docs
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload