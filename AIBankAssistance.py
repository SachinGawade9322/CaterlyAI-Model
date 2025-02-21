import os
import streamlit as st
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from PDF_Data_Extraction import TransactionProcessor
import re
from datetime import datetime

# Load the Fine-Tuned Financial Model
def load_finetuned_model():
    """Load the fine-tuned financial model."""
    MODEL_PATH = "bitext/Mistral-7B-Banking-v2"
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Extract Balance Information
def extract_balance(data):
    """Extract balance details dynamically and prevent errors."""
    balances = data.get("balances", {})
    return {
        "beginning_balance": float(balances.get("beginning_balance", "0").replace("$", "").replace(",", "")),
        "income_total": float(balances.get("income_total", "0").replace("$", "").replace(",", "")),
        "expense_total": float(balances.get("expense_total", "0").replace("$", "").replace(",", "")),
        "ending_balance": float(balances.get("ending_balance", "0").replace("$", "").replace(",", ""))
    }
    
def extract_transactions(data):
    """Extract transactions dynamically and categorize them properly."""
    transactions = data.get("transactions", {})
    
    extracted = {
        "income": [],
        "expenses": [],
        "atm_withdrawals": [],
        "credit_card_payments": []
    }

    for category, items in transactions.items():
        if not items:  
            continue

        for item in items:
            raw_amount = item.get("amount", "").replace("$", "").replace(",", "").strip()
            try:
                amount = float(raw_amount) if raw_amount else 0.0
            except ValueError:
                amount = 0.0

            company_name = item.get("details", {}).get("company_name", "").strip()

            transaction_entry = {
                "date": item.get("date", ""),
                "company_name": company_name,
                "amount": amount
            }

            # Categorization Logic
            if category == "Deposits and Additions":
                extracted["income"].append(transaction_entry)
            elif category == "ATM Withdrawals":
                extracted["atm_withdrawals"].append(transaction_entry)
            elif any(keyword in company_name for keyword in ["Card Purchase", "Payment", "Chase Card", "Zelle"]):
                extracted["credit_card_payments"].append(transaction_entry)
            else:
                extracted["expenses"].append(transaction_entry)

    return extracted

def summarize_transactions(user_query, transactions):
    """
    Generate a summary of transactions based on the user query.

    :param user_query: User's question (string)
    :param transactions: Extracted transaction data (dictionary)
    :return: Natural-language summary (string)
    """
    
    user_query = user_query.lower() 

    all_transactions = []
    for category, trans_list in transactions.items():
        if isinstance(trans_list, list):
            all_transactions.extend(trans_list)

    if not all_transactions:
        return "No transactions found in your records."

    first_match = re.search(r"first\s*(\d+)?", user_query)
    last_match = re.search(r"last\s*(\d+)?", user_query)
    company_match = re.search(r"(?:from|with|to)\s+([\w\s]+)", user_query)
    date_match = re.search(r"(\d{2}/\d{2})\s*(to|-)\s*(\d{2}/\d{2})?", user_query)

    extracted_transactions = []
    
    if first_match:
        count = int(first_match.group(1)) if first_match.group(1) else 1
        extracted_transactions = all_transactions[:count]
    elif last_match:
        count = int(last_match.group(1)) if last_match.group(1) else 1
        extracted_transactions = all_transactions[-count:]
    elif company_match:
        company = company_match.group(1).lower()
        extracted_transactions = [t for t in all_transactions if company in t["company_name"].lower()]
    elif date_match:
        start_date = date_match.group(1)
        end_date = date_match.group(3) if date_match.group(3) else start_date
        extracted_transactions = [t for t in all_transactions if start_date <= t["date"] <= end_date]

    if not extracted_transactions:
        return "No transactions matched your query."

    amounts = [t["amount"] for t in extracted_transactions]

    if "highest" in user_query or "maximum" in user_query or "max" in user_query:
        max_amount = max(amounts) if amounts else 0
        summary = f"The highest transaction in this selection is ${max_amount:.2f}."
    elif "lowest" in user_query or "minimum" in user_query or "min" in user_query:
        min_amount = min(amounts) if amounts else 0
        summary = f"The lowest transaction in this selection is ${min_amount:.2f}."
    elif "average" in user_query:
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        summary = f"The average spending in this selection is ${avg_amount:.2f}."
    elif "total spending" in user_query:
        total_spent = sum(a for a in amounts if a < 0)
        summary = f"The total spending in this selection is ${abs(total_spent):.2f}."
    elif "total income" in user_query:
        total_income = sum(a for a in amounts if a > 0)
        summary = f"The total income in this selection is ${total_income:.2f}."
    elif "total saving" in user_query:
        total_income = sum(a for a in amounts if a > 0)
        total_spent = sum(a for a in amounts if a < 0)
        total_savings = total_income + total_spent
        summary = f"Your total savings in this selection is ${total_savings:.2f}."
    else:
        summary = f"I found {len(extracted_transactions)} transactions matching your request. "
        for t in extracted_transactions[:5]: 
            summary += f"On {t['date']}, you made a transaction at {t['company_name']} for ${t['amount']:.2f}. "

    return summary

def forecast_future_balance(balance_info, months=6):
    """Predict the user's balance in the future based on current income and expenses."""
    monthly_savings = balance_info["income_total"] - balance_info["expense_total"]
    
    if monthly_savings < 0:
        return balance_info["ending_balance"] 
    
    future_balance = balance_info["ending_balance"] + (monthly_savings * months)
    return round(future_balance, 2)

# Investment Growth Calculation (Compound Interest)
def calculate_investment_growth(initial, rate, years):
    """Calculate future value of investment using compound interest."""
    future_value = initial * (1 + rate / 100) ** years
    return round(future_value, 2)

# Loan EMI Calculation
def calculate_loan_emi(principal, annual_rate, tenure_years):
    """Calculate EMI for a given loan."""
    monthly_rate = (annual_rate / 100) / 12
    months = tenure_years * 12

    if monthly_rate == 0:
        emi = principal / months  
    else:
        emi = (principal * monthly_rate) / (1 - (1 + monthly_rate) ** -months)

    total_payment = emi * months
    total_interest = total_payment - principal
    return {
        "monthly_emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2)
    }


# Debt-to-Income Ratio Calculation
def calculate_debt_to_income_ratio(debt, income):
    """Calculate the debt-to-income ratio (DTI)."""
    if income == 0:
        return "N/A"  # Avoid division by zero
    dti_ratio = (debt / income) * 100
    return round(dti_ratio, 2)

# Tax Estimation (Basic Calculation)
def estimate_tax(income, deductions=0):
    """Calculate tax with income deductions and proper tax brackets."""
    taxable_income = max(0, income - deductions) 

    tax_brackets = [
        (10000, 0.1),
        (40000, 0.15),
        (85000, 0.2),
        (float('inf'), 0.25)
    ]

    tax_payable = 0
    prev_limit = 0
    for limit, rate in tax_brackets:
        if taxable_income > prev_limit:
            taxable_at_this_rate = min(limit - prev_limit, taxable_income - prev_limit)
            tax_payable += taxable_at_this_rate * rate
        prev_limit = limit

    return round(tax_payable, 2)

def calculate_credit_utilization(transactions, credit_limit):
    """Calculate credit utilization ratio (CUR)."""
    total_credit_spent = sum(t["amount"] for t in transactions["credit_card_payments"])
    
    if credit_limit == 0:
        return "N/A" 
    
    utilization_ratio = (total_credit_spent / credit_limit) * 100
    return round(utilization_ratio, 2)

def recommend_emergency_fund(balance_info):
    """Recommend emergency savings based on monthly expenses."""
    avg_monthly_expense = balance_info["expense_total"] / 6
    return round(avg_monthly_expense * 3, 2) 

def get_transaction_extremes(transactions, date_range=None):
    """Find the highest and lowest transactions for full history or a given date range."""
    filtered_transactions = []
    
    for category, items in transactions.items():
        for item in items:
            if date_range:
                if item["date"] >= date_range[0] and item["date"] <= date_range[1]:
                    filtered_transactions.append(item)
            else:
                filtered_transactions.append(item)

    if not filtered_transactions:
        return None, None 

    highest_transaction = max(filtered_transactions, key=lambda x: x["amount"])
    lowest_transaction = min(filtered_transactions, key=lambda x: x["amount"])

    return highest_transaction, lowest_transaction

def calculate_transaction_averages(transactions, date_range=None):
    """Calculate the average transaction amount for full history or a date range."""
    filtered_transactions = []
    
    for category, items in transactions.items():
        for item in items:
            if date_range:
                if item["date"] >= date_range[0] and item["date"] <= date_range[1]:
                    filtered_transactions.append(item)
            else:
                filtered_transactions.append(item)

    if not filtered_transactions:
        return 0.0 

    total_amount = sum(t["amount"] for t in filtered_transactions)
    avg_amount = total_amount / len(filtered_transactions)

    return round(avg_amount, 2)

def categorize_expenses(transactions):
    """Categorize expenses by type."""
    category_totals = {}

    for expense in transactions["expenses"]:
        category = expense.get("company_name", "Other")
        category_totals[category] = category_totals.get(category, 0) + expense["amount"]

    return category_totals

def calculate_loan_balance(principal, annual_rate, tenure_years, months_paid):
    """Track remaining loan balance after making payments."""
    emi_data = calculate_loan_emi(principal, annual_rate, tenure_years)
    emi = emi_data["monthly_emi"]
    remaining_months = (tenure_years * 12) - months_paid
    remaining_balance = emi * remaining_months

    return round(remaining_balance, 2)

def calculate_credit_card_interest(balance, apr, min_payment_rate=0.02):
    """Calculate interest on outstanding credit card balance."""
    monthly_interest = (apr / 100) / 12
    interest_charged = balance * monthly_interest
    min_payment = max(balance * min_payment_rate, 25)  

    return {
        "interest_charged": round(interest_charged, 2),
        "minimum_payment": round(min_payment, 2),
        "new_balance": round(balance + interest_charged, 2)
    }

def generate_dynamic_context(data, user_query):
    """Create structured financial summary and dynamically summarize transactions based on user query."""

    balance_info = extract_balance(data)
    transactions = extract_transactions(data)

    extracted_summary = summarize_transactions(user_query, transactions)

    highest_transaction, lowest_transaction = get_transaction_extremes(transactions)
    avg_transaction = calculate_transaction_averages(transactions)
    future_balance_6m = forecast_future_balance(balance_info, months=6)
    estimated_tax = estimate_tax(balance_info["income_total"])
    credit_utilization = calculate_credit_utilization(transactions, credit_limit=5000)
    emergency_fund = recommend_emergency_fund(balance_info)

    loan_balance = calculate_loan_balance(principal=50000, annual_rate=5, tenure_years=10, months_paid=24)
    credit_card_interest = calculate_credit_card_interest(balance=3000, apr=18)

    expense_categories = categorize_expenses(transactions)
    category_summary = "\n".join([f"- {cat}: ${amount:,.2f}" for cat, amount in expense_categories.items()])

    context = f"""
    🔹 **Financial Summary:**
    - Beginning Balance: **${balance_info['beginning_balance']:,.2f}**
    - Total Income: **${balance_info['income_total']:,.2f}**
    - Total Expenses: **${balance_info['expense_total']:,.2f}**
    - Ending Balance: **${balance_info['ending_balance']:,.2f}**
    - **Projected Balance (6 months)**: **${future_balance_6m:,.2f}**
    - **Estimated Tax**: **${estimated_tax:,.2f}**
    - **Credit Utilization Ratio**: **{credit_utilization}%**
    - **Recommended Emergency Fund**: **${emergency_fund:,.2f}**
    
    🔹 **Loan & Credit Card Summary:**
    - **Remaining Loan Balance**: **${loan_balance:,.2f}**
    - **Credit Card Interest Charged**: **${credit_card_interest['interest_charged']:,.2f}**
    - **Minimum Payment Due**: **${credit_card_interest['minimum_payment']:,.2f}**
    - **New Credit Card Balance**: **${credit_card_interest['new_balance']:,.2f}**

    🔹 **Transactions Breakdown:**
    - **Highest Transaction**: {highest_transaction['company_name']} (**${highest_transaction['amount']:,.2f}**) if highest_transaction else "N/A"
    - **Lowest Transaction**: {lowest_transaction['company_name']} (**${lowest_transaction['amount']:,.2f}**) if lowest_transaction else "N/A"
    - **Average Transaction Amount**: **${avg_transaction:,.2f}**
    
    🔹 **Category-wise Expense Summary:**
    {category_summary}

    🔹 **Income Sources:**
    {', '.join([f"{t['company_name']} (**${t['amount']:,.2f}**)" for t in transactions['income']])}

    🔹 **Major Expenses:**
    {', '.join([f"{t['company_name']} (**${t['amount']:,.2f}**)" for t in transactions['expenses']])}

    🔹 **ATM Withdrawals:**
    {', '.join([f"{t['date']} (**${t['amount']:,.2f}**)" for t in transactions['atm_withdrawals']])}

    🔹 **🔍 User Query Response:**
    {extracted_summary}  # Dynamically added transaction summary
    """
    
    return context

def replace_placeholders(response, balance_info, transactions, user_query):
    """Replace placeholders dynamically with real financial values."""

    highest_transaction, lowest_transaction = get_transaction_extremes(transactions) if transactions else (None, None)

    future_balance_6m = forecast_future_balance(balance_info, months=6)
    estimated_tax = estimate_tax(balance_info["income_total"])
    credit_utilization = calculate_credit_utilization(transactions, credit_limit=5000)
    emergency_fund = recommend_emergency_fund(balance_info)

    loan_balance = calculate_loan_balance(principal=50000, annual_rate=5, tenure_years=10, months_paid=24) 
    credit_card_interest = calculate_credit_card_interest(balance=3000, apr=18) 

    expense_categories = categorize_expenses(transactions)
    category_summary = "\n".join([f"- {cat}: **${amount:,.2f}**" for cat, amount in expense_categories.items()]) if expense_categories else "N/A"

    transaction_summary = summarize_transactions(user_query, transactions)

    placeholders = {
        "{{total_income}}": f"${balance_info['income_total']:,.2f}",
        "{{total_expense}}": f"${balance_info['expense_total']:,.2f}",
        "{{ending_balance}}": f"${balance_info['ending_balance']:,.2f}",
        "{{projected_balance}}": f"${future_balance_6m:,.2f}",
        "{{estimated_tax}}": f"${estimated_tax:,.2f}",
        "{{credit_utilization}}": f"{credit_utilization}%",
        "{{recommended_emergency_fund}}": f"${emergency_fund:,.2f}",
        "{{expense_breakdown}}": category_summary,
        "{{atm_withdrawals}}": f"${sum(t['amount'] for t in transactions.get('atm_withdrawals', []) if isinstance(t, dict) and 'amount' in t):,.2f}" if isinstance(transactions.get("atm_withdrawals"), list) else "N/A",
        "{{highest_transaction}}": f"{highest_transaction['company_name']} (${highest_transaction['amount']:,.2f})" if highest_transaction else "N/A",
        "{{lowest_transaction}}": f"{lowest_transaction['company_name']} (${lowest_transaction['amount']:,.2f})" if lowest_transaction else "N/A",
        "{{loan_balance}}": f"${loan_balance:,.2f}",
        "{{credit_card_interest_charged}}": f"${credit_card_interest['interest_charged']:,.2f}",
        "{{credit_card_minimum_payment}}": f"${credit_card_interest['minimum_payment']:,.2f}",
        "{{credit_card_new_balance}}": f"${credit_card_interest['new_balance']:,.2f}",
        "{{transaction_summary}}": transaction_summary
    }

    for placeholder, value in placeholders.items():
        response = response.replace(placeholder, value)

    return response

def generate_response(model, tokenizer, user_query, context, balance_info, transactions):
    """Generate highly detailed, structured, and informative financial advice with dynamic calculations."""
    
    if not model or not tokenizer:
        return "Error: Model not loaded."

    input_text = f"""
        You are an expert financial assistant. Provide long, highly detailed financial advice with dynamic calculations.
        
        - Start by analyzing the user's financial status based on income, expenses, and transaction history.
        - Identify financial trends, such as increased spending patterns, savings performance, and credit utilization.
        - Provide structured insights, including spending breakdowns, budgeting recommendations, and risk management strategies.
        - Offer detailed step-by-step calculations for taxes, loans, savings forecasts, and debt analysis.
        - If the query relates to transactions, show the highest/lowest expenses, categorized breakdowns, and financial impact.
        - If the user asks about future savings, predict financial trends based on past behavior.
        - Provide structured tables, bullet points, and comparisons for clear readability.
        - Avoid generic responses; ensure every response is personalized to the user’s financial data.
        - Summarize key financial recommendations at the end of the response.

         **Response should be:** 
        - **Long, highly detailed, and informative**
        - **Descriptive financial advice with proper explanations**
        - **Accurate, structured, and human-like insights**

        Context:
        {context}

        User Query: {user_query}

        Assistant:
    """

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

    output = model.generate(
        **inputs,
        max_new_tokens=3000, 
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return replace_placeholders(response, balance_info, transactions, user_query)

def main():
    st.title("💰 AI-Powered Financial Assistant")

    if "model" not in st.session_state:
        with st.spinner("Loading AI Model... Please wait..."):
            st.session_state.model, st.session_state.tokenizer = load_finetuned_model()
        st.success("Model Loaded Successfully!")

    uploaded_file = st.file_uploader("📂 Upload Your Bank Statement", type=["pdf", "png", "jpg", "jpeg"])
    user_query = st.text_input("📝 Ask your financial question:")
    submit_button = st.button("Submit")

    if submit_button and uploaded_file and user_query:
        try:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            processor = TransactionProcessor()
            json_data = processor.process_file(temp_file_path)

            output_path = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output\extracted_transactions.json"
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4)

            if not json_data or "transactions" not in json_data:
                st.error("No transactions found. Verify the extracted text and regex patterns.")
                return 

            balance_info = extract_balance(json_data)
            transactions = extract_transactions(json_data)
            context = generate_dynamic_context(json_data, user_query) 

            model, tokenizer = st.session_state.model, st.session_state.tokenizer
            if model and tokenizer:
                response = generate_response(model, tokenizer, user_query, context, balance_info, transactions)
                st.write("### Response:")
                st.write(response)
            else:
                st.error("Model failed to load. Please check your setup.")

            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()