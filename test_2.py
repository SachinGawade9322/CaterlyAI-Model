# import streamlit as st
# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import numpy as np

# # ✅ Load the Fine-Tuned Financial Model
# def load_finetuned_model():
#     """Load the fine-tuned financial model."""
#     MODEL_PATH = "bitext/Mistral-7B-Banking-v2"
#     try:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )

#         tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#         tokenizer.pad_token = tokenizer.eos_token

#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             device_map="cuda:0",
#             quantization_config=quantization_config,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )
#         model.eval()
#         return model, tokenizer
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None

# # ✅ Extract Balance Information
# def extract_balance(data):
#     """Extract balance details dynamically and prevent errors."""
#     balances = data.get("balances", {})
#     return {
#         "beginning_balance": float(balances.get("beginning_balance", "0").replace("$", "").replace(",", "")),
#         "income_total": float(balances.get("income_total", "0").replace("$", "").replace(",", "")),
#         "expense_total": float(balances.get("expense_total", "0").replace("$", "").replace(",", "")),
#         "ending_balance": float(balances.get("ending_balance", "0").replace("$", "").replace(",", ""))
#     }
    
# # ✅ Extract Transactions & Categorize
# def extract_transactions(data):
#     """Extract transactions dynamically and categorize them properly."""
#     transactions = data.get("transactions", {})
#     extracted = {"income": [], "expenses": [], "atm_withdrawals": [], "credit_card_payments": []}

#     for category, items in transactions.items():
#         for item in items:
#             raw_amount = item.get("amount", "").replace("$", "").replace(",", "")
#             try:
#                 amount = float(raw_amount) if raw_amount else 0.0
#             except ValueError:
#                 amount = 0.0

#             transaction_entry = {
#                 "date": item.get("date", ""),
#                 "company_name": item.get("details", {}).get("company_name", ""),
#                 "amount": amount
#             }

#             # Categorize Transactions
#             if category == "Deposits and Additions":
#                 extracted["income"].append(transaction_entry)
#             elif category == "Withdrawals and Payments":
#                 if "ATM Withdrawal" in transaction_entry["company_name"]:
#                     extracted["atm_withdrawals"].append(transaction_entry)
#                 elif "Payment" in transaction_entry["company_name"]:
#                     extracted["credit_card_payments"].append(transaction_entry)
#                 else:
#                     extracted["expenses"].append(transaction_entry)
#             else:
#                 extracted["expenses"].append(transaction_entry)

#     return extracted


# # ✅ Forecast Future Balance Based on Spending Trends
# def forecast_future_balance(balance_info, months=6):
#     """Predict the user's balance in the future based on current income and expenses."""
#     monthly_savings = balance_info["income_total"] - balance_info["expense_total"]
#     future_balance = balance_info["ending_balance"] + (monthly_savings * months)
#     return round(future_balance, 2)

# # ✅ Investment Growth Calculation (Compound Interest)
# def calculate_investment_growth(initial, rate, years):
#     """Calculate future value of investment using compound interest."""
#     future_value = initial * (1 + rate / 100) ** years
#     return round(future_value, 2)

# # ✅ Loan EMI Calculation
# def calculate_loan_emi(principal, annual_rate, tenure_years):
#     """Calculate EMI for a given loan."""
#     monthly_rate = (annual_rate / 100) / 12
#     months = tenure_years * 12
#     if monthly_rate == 0:
#         emi = principal / months  # In case of 0% interest
#     else:
#         emi = (principal * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
#     total_payment = emi * months
#     total_interest = total_payment - principal
#     return {
#         "monthly_emi": round(emi, 2),
#         "total_payment": round(total_payment, 2),
#         "total_interest": round(total_interest, 2)
#     }

# # ✅ Debt-to-Income Ratio Calculation
# def calculate_debt_to_income_ratio(debt, income):
#     """Calculate the debt-to-income ratio (DTI)."""
#     if income == 0:
#         return "N/A"  # Avoid division by zero
#     dti_ratio = (debt / income) * 100
#     return round(dti_ratio, 2)

# # ✅ Tax Estimation (Basic Calculation)
# def estimate_tax(income):
#     """Estimate tax payable based on income brackets."""
#     if income <= 10000:
#         tax_rate = 10
#     elif income <= 40000:
#         tax_rate = 15
#     elif income <= 85000:
#         tax_rate = 20
#     else:
#         tax_rate = 25

#     tax_payable = (income * tax_rate) / 100
#     return round(tax_payable, 2)

# # ✅ Highest & Lowest Transactions (Full Statement or Date Range)
# def get_transaction_extremes(transactions, date_range=None):
#     """Find the highest and lowest transactions for full history or a given date range."""
#     filtered_transactions = []
    
#     for category, items in transactions.items():
#         for item in items:
#             if date_range:
#                 if item["date"] >= date_range[0] and item["date"] <= date_range[1]:
#                     filtered_transactions.append(item)
#             else:
#                 filtered_transactions.append(item)

#     if not filtered_transactions:
#         return None, None  # No transactions found

#     highest_transaction = max(filtered_transactions, key=lambda x: x["amount"])
#     lowest_transaction = min(filtered_transactions, key=lambda x: x["amount"])

#     return highest_transaction, lowest_transaction

# # ✅ Compute Averages for Transactions
# def calculate_transaction_averages(transactions, date_range=None):
#     """Calculate the average transaction amount for full history or a date range."""
#     filtered_transactions = []
    
#     for category, items in transactions.items():
#         for item in items:
#             if date_range:
#                 if item["date"] >= date_range[0] and item["date"] <= date_range[1]:
#                     filtered_transactions.append(item)
#             else:
#                 filtered_transactions.append(item)

#     if not filtered_transactions:
#         return 0.0  # No transactions found

#     total_amount = sum(t["amount"] for t in filtered_transactions)
#     avg_amount = total_amount / len(filtered_transactions)

#     return round(avg_amount, 2)

# # ✅ Generate Structured Financial Context for Model
# def generate_dynamic_context(data):
#     """Create structured financial summary for better AI understanding."""
#     balance_info = extract_balance(data)
#     transactions = extract_transactions(data)

#     highest_transaction, lowest_transaction = get_transaction_extremes(transactions)
#     avg_transaction = calculate_transaction_averages(transactions)
#     future_balance_6m = forecast_future_balance(balance_info, months=6)
#     estimated_tax = estimate_tax(balance_info["income_total"])

#     context = f"""
#     🔹 Financial Summary:
#     - Beginning Balance: ${balance_info['beginning_balance']:,.2f}
#     - Total Income: ${balance_info['income_total']:,.2f}
#     - Total Expenses: ${balance_info['expense_total']:,.2f}
#     - Ending Balance: ${balance_info['ending_balance']:,.2f}
#     - Projected Balance (6 months): ${future_balance_6m:,.2f}
#     - Estimated Tax: ${estimated_tax:,.2f}

#     🔹 Transactions Breakdown:
#     - Highest Transaction: {highest_transaction['company_name']} (${highest_transaction['amount']:,.2f}) if highest_transaction else "N/A"
#     - Lowest Transaction: {lowest_transaction['company_name']} (${lowest_transaction['amount']:,.2f}) if lowest_transaction else "N/A"
#     - Average Transaction Amount: ${avg_transaction:,.2f}

#     - Income Sources: {', '.join([f"{t['company_name']} (${t['amount']:,.2f})" for t in transactions['income']])}
#     - Major Expenses: {', '.join([f"{t['company_name']} (${t['amount']:,.2f})" for t in transactions['expenses']])}
#     - ATM Withdrawals: {', '.join([f"{t['date']} (${t['amount']:,.2f})" for t in transactions['atm_withdrawals']])}
#     """
#     return context

# # ✅ Fix Placeholders in Response
# def replace_placeholders(response, balance_info, transactions):
#     """Replace placeholders dynamically with real values."""
#     highest_transaction, lowest_transaction = get_transaction_extremes(transactions)
#     future_balance_6m = forecast_future_balance(balance_info, months=6)
#     estimated_tax = estimate_tax(balance_info["income_total"])

#     placeholders = {
#         "{{total_income}}": f"${balance_info['income_total']:,.2f}",
#         "{{total_expense}}": f"${balance_info['expense_total']:,.2f}",
#         "{{ending_balance}}": f"${balance_info['ending_balance']:,.2f}",
#         "{{projected_balance}}": f"${future_balance_6m:,.2f}",
#         "{{estimated_tax}}": f"${estimated_tax:,.2f}",
#         "{{atm_withdrawals}}": f"${sum(t['amount'] for t in transactions['atm_withdrawals']):,.2f}",
#         "{{highest_transaction}}": f"{highest_transaction['company_name']} (${highest_transaction['amount']:,.2f})" if highest_transaction else "N/A",
#         "{{lowest_transaction}}": f"{lowest_transaction['company_name']} (${lowest_transaction['amount']:,.2f})" if lowest_transaction else "N/A"
#     }
    
#     for placeholder, value in placeholders.items():
#         response = response.replace(placeholder, value)

#     return response

# # ✅ Generate AI-Powered Response
# def generate_response(model, tokenizer, user_query, context, balance_info, transactions):
#     """Generate human-like financial advice with dynamic calculations."""
#     if not model or not tokenizer:
#         return "Error: Model not loaded."

#     input_text = f"""
#         You are an expert financial assistant. Provide detailed step-by-step calculations, financial insights, and structured responses.
        
#         - Always provide step-by-step calculations with proper financial insights.
#         - Ensure responses include clear, structured tables and detailed analysis.
#         - Avoid static responses; dynamically generate explanations based on data.
#         - Format all currency values correctly (X,XXX.XX).
#         - Summarize transactions in an easy-to-read manner.
#         - Provide long, descriptive, and informative financial advice.
#         - Offer recommendations based on user spending and investment patterns.

#         Context:
#         {context}

#         User Query: {user_query}
#         Assistant:
#     """

#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

#     output = model.generate(
#         **inputs,
#         max_new_tokens=1800,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9
#     )

#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return replace_placeholders(response, balance_info, transactions)

# # ✅ Streamlit UI for Chatbot
# st.title("💰 AI-Powered Financial Assistant")

# # ✅ Upload Transaction JSON File
# uploaded_file = st.file_uploader("📂 Upload Transactions JSON", type=["json"])
# user_query = st.text_input("📝 Ask your financial question:")
# submit_button = st.button("Submit")

# if submit_button and uploaded_file and user_query:
#     try:
#         json_data = json.load(uploaded_file)
#         balance_info = extract_balance(json_data)
#         transactions = extract_transactions(json_data)
#         context = generate_dynamic_context(json_data)

#         # ✅ Load Model & Generate Response
#         model, tokenizer = load_finetuned_model()
#         if model and tokenizer:
#             response = generate_response(model, tokenizer, user_query, context, balance_info, transactions)
#             st.write("### 📊 Response:")
#             st.write(response)
#         else:
#             st.error("❌ Model failed to load. Please check your setup.")

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")






import os
import streamlit as st
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from pdf_to_json import process_bank_statement

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
    
# Extract Transactions & Categorize (Unchanged)
# def extract_transactions(data):
#     """Extract transactions dynamically and categorize them properly."""
#     transactions = data.get("transactions", {})
#     extracted = {"income": [], "expenses": [], "atm_withdrawals": [], "credit_card_payments": []}

#     for category, items in transactions.items():
#         for item in items:
#             raw_amount = item.get("amount", "").replace("$", "").replace(",", "")
#             try:
#                 amount = float(raw_amount) if raw_amount else 0.0
#             except ValueError:
#                 amount = 0.0

#             transaction_entry = {
#                 "date": item.get("date", ""),
#                 "company_name": item.get("details", {}).get("company_name", ""),
#                 "amount": amount
#             }

#             if category == "Deposits and Additions":
#                 extracted["income"].append(transaction_entry)
#             elif category == "Withdrawals and Payments":
#                 if "ATM Withdrawal" in transaction_entry["company_name"]:
#                     extracted["atm_withdrawals"].append(transaction_entry)
#                 elif "Payment" in transaction_entry["company_name"]:
#                     extracted["credit_card_payments"].append(transaction_entry)
#                 else:
#                     extracted["expenses"].append(transaction_entry)
#             else:
#                 extracted["expenses"].append(transaction_entry)

#     return extracted

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
        if not items:  # Skip if category is null
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

def generate_dynamic_context(data):
    """Create structured financial summary for better AI understanding."""

    balance_info = extract_balance(data)
    transactions = extract_transactions(data)

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
    """
    return context

def replace_placeholders(response, balance_info, transactions):
    """Replace placeholders dynamically with real financial values."""

    # Get highest & lowest transactions (handle missing transactions)
    highest_transaction, lowest_transaction = get_transaction_extremes(transactions) if transactions else (None, None)

    # Compute future balance, tax estimate, and other key financial insights
    future_balance_6m = forecast_future_balance(balance_info, months=6)
    estimated_tax = estimate_tax(balance_info["income_total"])
    credit_utilization = calculate_credit_utilization(transactions, credit_limit=5000)
    emergency_fund = recommend_emergency_fund(balance_info)

    # Loan and Credit Card Calculations
    loan_balance = calculate_loan_balance(principal=50000, annual_rate=5, tenure_years=10, months_paid=24)  # Example Loan
    credit_card_interest = calculate_credit_card_interest(balance=3000, apr=18)  # Example Credit Card

    # Categorize expenses for breakdown
    expense_categories = categorize_expenses(transactions)
    category_summary = "\n".join([f"- {cat}: **${amount:,.2f}**" for cat, amount in expense_categories.items()]) if expense_categories else "N/A"

    # Define placeholders dynamically
    placeholders = {
        "{{total_income}}": f"${balance_info['income_total']:,.2f}",
        "{{total_expense}}": f"${balance_info['expense_total']:,.2f}",
        "{{ending_balance}}": f"${balance_info['ending_balance']:,.2f}",
        "{{projected_balance}}": f"${future_balance_6m:,.2f}",
        "{{estimated_tax}}": f"${estimated_tax:,.2f}",
        "{{credit_utilization}}": f"{credit_utilization}%",
        "{{recommended_emergency_fund}}": f"${emergency_fund:,.2f}",
        "{{expense_breakdown}}": category_summary,
        "{{atm_withdrawals}}": f"${sum(t['amount'] for t in transactions['atm_withdrawals']):,.2f}" if transactions.get("atm_withdrawals") else "N/A",
        "{{highest_transaction}}": f"{highest_transaction['company_name']} (${highest_transaction['amount']:,.2f})" if highest_transaction else "N/A",
        "{{lowest_transaction}}": f"{lowest_transaction['company_name']} (${lowest_transaction['amount']:,.2f})" if lowest_transaction else "N/A",
        "{{loan_balance}}": f"${loan_balance:,.2f}",
        "{{credit_card_interest_charged}}": f"${credit_card_interest['interest_charged']:,.2f}",
        "{{credit_card_minimum_payment}}": f"${credit_card_interest['minimum_payment']:,.2f}",
        "{{credit_card_new_balance}}": f"${credit_card_interest['new_balance']:,.2f}"
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

        📌 **Response should be:** 
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
    return replace_placeholders(response, balance_info, transactions)

# # ✅ Streamlit UI for Chatbot
# st.title("💰 AI-Powered Financial Assistant")

# uploaded_file = st.file_uploader("📂 Upload Transactions JSON", type=["json"])
# user_query = st.text_input("📝 Ask your financial question:")
# submit_button = st.button("Submit")

# if submit_button and uploaded_file and user_query:
#     try:
#         json_data = json.load(uploaded_file)
#         balance_info = extract_balance(json_data)
#         transactions = extract_transactions(json_data)
#         context = generate_dynamic_context(json_data)

#         model, tokenizer = load_finetuned_model()
#         if model and tokenizer:
#             response = generate_response(model, tokenizer, user_query, context, balance_info, transactions)
#             st.write("### 📊 Response:")
#             st.write(response)
#         else:
#             st.error("❌ Model failed to load. Please check your setup.")

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")


# def main():
#     st.title("💰 AI-Powered Financial Assistant")
    
#     model, tokenizer = load_finetuned_model()

#     uploaded_file = st.file_uploader("📂 Upload Your Bank Statement", type=["pdf", "png", "jpg", "jpeg"])
#     user_query = st.text_input("📝 Ask your financial question:")
#     submit_button = st.button("Submit")

#     if submit_button and uploaded_file and user_query:
#         try:
#             temp_file_path = f"temp_{uploaded_file.name}"
#             with open(temp_file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             process_bank_statement(temp_file_path)

#             extracted_json_path = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output\extracted_transactions.json"
#             if os.path.exists(extracted_json_path):
#                 with open(extracted_json_path, "r") as json_file:
#                     json_data = json.load(json_file)
#             else:
#                 st.error("❌ Extraction failed. No transaction data found.")
#                 return 

#             balance_info = extract_balance(json_data)
#             transactions = extract_transactions(json_data)
#             context = generate_dynamic_context(json_data)

#             # model, tokenizer = load_finetuned_model()
#             if model and tokenizer:
#                 response = generate_response(model, tokenizer, user_query, context, balance_info, transactions)
#                 st.write("### 📊 Response:")
#                 st.write(response)
#             else:
#                 st.error("❌ Model failed to load. Please check your setup.")

#             os.remove(temp_file_path)

#         except Exception as e:
#             st.error(f"❌ Error processing file: {e}")

# if __name__ == "__main__":
#     main()


def main():
    st.title("💰 AI-Powered Financial Assistant")

    if "model" not in st.session_state:
        with st.spinner("🔄 Loading AI Model... Please wait..."):
            st.session_state.model, st.session_state.tokenizer = load_finetuned_model()
        st.success("✅ Model Loaded Successfully!")

    uploaded_file = st.file_uploader("📂 Upload Your Bank Statement", type=["pdf", "png", "jpg", "jpeg"])
    user_query = st.text_input("📝 Ask your financial question:")
    submit_button = st.button("Submit")

    if submit_button and uploaded_file and user_query:
        try:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            process_bank_statement(temp_file_path)

            extracted_json_path = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output\extracted_transactions.json"
            if os.path.exists(extracted_json_path):
                with open(extracted_json_path, "r") as json_file:
                    json_data = json.load(json_file)
            else:
                st.error("❌ Extraction failed. No transaction data found.")
                return 

            balance_info = extract_balance(json_data)
            transactions = extract_transactions(json_data)
            context = generate_dynamic_context(json_data)

            model, tokenizer = st.session_state.model, st.session_state.tokenizer
            if model and tokenizer:
                response = generate_response(model, tokenizer, user_query, context, balance_info, transactions)
                st.write("### 📊 Response:")
                st.write(response)
            else:
                st.error("❌ Model failed to load. Please check your setup.")

            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()
