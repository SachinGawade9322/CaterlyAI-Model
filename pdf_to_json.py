# # import pytesseract
# # import cv2
# # import re
# # import fitz  # PyMuPDF for handling PDFs
# # import os
# # import json
# # from pdf2image import convert_from_path

# # # ✅ Extract text from an image (PNG, JPG, JPEG)
# # def extract_text_from_image(image_path):
# #     """Extracts text from an image file using OCR."""
# #     try:
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             print(f"❌ Error: Unable to read image at {image_path}")
# #             return ""

# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #         text = pytesseract.image_to_string(gray)
# #         return text
# #     except Exception as e:
# #         print(f"❌ Error during OCR: {e}")
# #         return ""

# # # ✅ Extract text from a PDF
# # def extract_text_from_pdf(pdf_path):
# #     """Extracts text from a PDF file using OCR."""
# #     text = ""
# #     try:
# #         images = convert_from_path(pdf_path)
# #         for i, image in enumerate(images):
# #             temp_image_path = f"page_{i}.jpg"
# #             image.save(temp_image_path, "JPEG")
# #             text += extract_text_from_image(temp_image_path)
# #             os.remove(temp_image_path)  # Clean up temp files
# #         return text
# #     except Exception as e:
# #         print(f"❌ Error processing PDF: {e}")
# #         return ""

# # # ✅ Extract transactions from OCR text
# # def extract_transaction_data(text):
# #     """Extracts and categorizes transaction data from OCR text."""
# #     print("\n📌 Raw OCR Text Extracted:")
# #     print(text)

# #     # ✅ Extract balances dynamically
# #     balance_pattern = r"(Beginning Balance|Ending Balance):\s*\$([\d,]+\.\d{2})"
# #     balance_matches = re.findall(balance_pattern, text)

# #     balances = {
# #         "beginning_balance": "$0.00",
# #         "income_total": "$0.00",
# #         "expense_total": "$0.00",
# #         "ending_balance": "$0.00"
# #     }

# #     for match in balance_matches:
# #         label, amount = match
# #         if "Beginning Balance" in label:
# #             balances["beginning_balance"] = f"${amount}"
# #         elif "Ending Balance" in label:
# #             balances["ending_balance"] = f"${amount}"

# #     # ✅ Extract transactions
# #     pattern = r'(\d{2}/\d{2})\s+(.*?)\s+([\d,]+\.\d{2})'
# #     matches = re.findall(pattern, text)

# #     deposits = []
# #     atm_withdrawals = []
# #     electronic_withdrawals = []
# #     other_withdrawals = []

# #     for match in matches:
# #         date, description, amount = match
# #         transaction = {
# #             "date": date,
# #             "details": {
# #                 "company_name": description.strip(),
# #                 "origin_id": "",
# #                 "date_description": "CO Entry",
# #                 "name": description.split()[0] if description else "",
# #                 "id": ""
# #             },
# #             "amount": f"${amount}"
# #         }

# #         # Categorize transactions based on keywords
# #         if "deposit" in description.lower() or "credit" in description.lower():
# #             deposits.append(transaction)
# #         elif "atm" in description.lower():
# #             atm_withdrawals.append(transaction)
# #         elif "transfer" in description.lower() or "electronic" in description.lower():
# #             electronic_withdrawals.append(transaction)
# #         else:
# #             other_withdrawals.append(transaction)

# #     # ✅ Compute total income & expenses dynamically
# #     total_income = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in deposits) if deposits else 0.00
# #     total_expense = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in atm_withdrawals + electronic_withdrawals + other_withdrawals) if (atm_withdrawals + electronic_withdrawals + other_withdrawals) else 0.00

# #     balances["income_total"] = f"${total_income:,.2f}"
# #     balances["expense_total"] = f"${total_expense:,.2f}"

# #     # ✅ Structured JSON output
# #     transactions_json = {
# #         "transactions": {
# #             "Deposits and Additions": deposits if deposits else None,
# #             "ATM Withdrawals": atm_withdrawals if atm_withdrawals else None,
# #             "Electronic Withdrawals": electronic_withdrawals if electronic_withdrawals else None,
# #             "Other Withdrawals": other_withdrawals if other_withdrawals else None
# #         },
# #         "balances": balances
# #     }

# #     return transactions_json

# # # ✅ Process uploaded file (PDF or Image)
# # def process_bank_statement(file_path):
# #     """Extract transactions from a bank statement (PDF or image)."""
# #     if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
# #         extracted_text = extract_text_from_image(file_path)
# #     elif file_path.lower().endswith('.pdf'):
# #         extracted_text = extract_text_from_pdf(file_path)
# #     else:
# #         print("❌ Unsupported file format. Please upload a PNG, JPG, or PDF file.")
# #         return

# #     # Extract transactions from OCR text
# #     transactions_json = extract_transaction_data(extracted_text)

# #     # Save to JSON file
# #     output_folder = "C:\\Users\\Admin\\Desktop\\Banking_chatbot\\bankingproject\\json_output"
# #     os.makedirs(output_folder, exist_ok=True)  # ✅ Creates folder if it doesn't exist

# #     output_file = os.path.join(output_folder, "extracted_transactions.json")

# #     with open(output_file, "w") as json_file:
# #         json.dump(transactions_json, json_file, indent=4)

# #     print(f"\n✅ Transactions extracted and saved to `{output_file}`")


# # # ✅ Example Usage (Replace with actual file path)
# # # process_bank_statement("your_bank_statement.pdf")
# # #   C:\Program Files\Tesseract-OCR




# # Step 2: Import Required Libraries (No changes needed)
# import pytesseract
# import cv2
# import re
# import fitz  # PyMuPDF for handling PDFs
# import os
# from pdf2image import convert_from_path
# import json  # For JSON formatting

# # Step 3: Define Function for OCR Processing (No changes needed)
# def extract_text_from_image(image_path):
#     """Extracts text from an image file using OCR."""
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Error: Unable to read image at {image_path}")
#             return ""

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         text = pytesseract.image_to_string(gray)
#         return text
#     except Exception as e:
#         print(f"Error during OCR: {e}")
#         return ""

# # Step 4: Define Function to Extract Transaction Data (Updated)
# def extract_transaction_data(text):
#     """
#     Extracts transaction data from OCR text and categorizes it into:
#     - Deposits and Additions
#     - ATM & DEBIT CARD WITHDRAWALS
#     - ELECTRONIC WITHDRAWALS
#     - OTHER WITHDRAWALS
#     Sets categories to None if they are empty.
#     """
#     print("\nRaw OCR Text:")
#     print(text)
#     print("\nExtracting data...\n")

#     # Regex pattern to match transaction data
#     pattern = r'(\d{2}/\d{2})\s+(.*?)\s+([\d,]+\.\d{2})'
#     matches = re.findall(pattern, text)

#     deposits = []
#     atm_debit_withdrawals = []
#     electronic_withdrawals = []
#     other_withdrawals = []

#     for match in matches:
#         date, description, amount = match

#         # Clean up the description to remove unwanted data (e.g., amounts, dates)
#         cleaned_description = re.sub(r'\d{2}/\d{2}', '', description)  # Remove dates
#         cleaned_description = re.sub(r'[\d,]+\.\d{2}', '', cleaned_description)  # Remove amounts
#         cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()  # Remove extra spaces

#         # Ensure all attributes are present, even if empty
#         transaction = {
#             "date": date,
#             "details": {
#                 "company_name": cleaned_description if cleaned_description else "",
#                 "origin_id": "",  # Empty by default
#                 "date_description": "CO Entry",  # Default value
#                 "name": cleaned_description.split()[0] if cleaned_description else "",  # First word of cleaned description
#                 "id": ""  # Empty by default
#             },
#             "amount": f"${amount}"
#         }

#         # Categorize transactions based on keywords
#         if "deposit" in cleaned_description.lower() or "credit" in cleaned_description.lower():
#             deposits.append(transaction)
#         elif "atm" in cleaned_description.lower() or "debit" in cleaned_description.lower():
#             atm_debit_withdrawals.append(transaction)
#         elif "transfer" in cleaned_description.lower() or "electronic" in cleaned_description.lower():
#             electronic_withdrawals.append(transaction)
#         else:
#             other_withdrawals.append(transaction)

#     # Compute total income and expenses dynamically
#     total_income = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in deposits) if deposits else 0.00
#     total_expense = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in atm_debit_withdrawals + electronic_withdrawals + other_withdrawals) if (atm_debit_withdrawals + electronic_withdrawals + other_withdrawals) else 0.00

#     # Build the final JSON structure with categories set to None if empty
#     transactions_json = {
#         "transactions": {
#             "Deposits and Additions": deposits if deposits else None,
#             "ATM & DEBIT CARD WITHDRAWALS": atm_debit_withdrawals if atm_debit_withdrawals else None,
#             "ELECTRONIC WITHDRAWALS": electronic_withdrawals if electronic_withdrawals else None,
#             "OTHER WITHDRAWALS": other_withdrawals if other_withdrawals else None
#         },
#         "check_summary": None,
#         "balances": {
#             "beginning_balance": "$0.00",  # Placeholder (can be updated dynamically if data is available)
#             "income_total": f"${total_income:,.2f}",  # Dynamically computed
#             "expense_total": f"${total_expense:,.2f}",  # Dynamically computed
#             "ending_balance": "$0.00"  # Placeholder (can be updated dynamically if data is available)
#         }
#     }

#     return transactions_json

# # Step 5: Define Function to Extract Text from PDF (No changes needed)
# def extract_text_from_pdf(pdf_path):
#     """Extracts text from a PDF file using OCR."""
#     text = ""
#     try:
#         images = convert_from_path(pdf_path)
#         for i, image in enumerate(images):
#             temp_image_path = f"page_{i}.jpg"
#             image.save(temp_image_path, "JPEG")
#             text += extract_text_from_image(temp_image_path)
#             os.remove(temp_image_path)  # Clean up temp files
#         return text
#     except Exception as e:
#         print(f"Error processing PDF: {e}")
#         return ""

# # Step 6: Upload and Process Files (No changes needed)
# uploaded = files.upload()

# for filename in uploaded.keys():
#     print(f"\nProcessing file: {filename}")

#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         extracted_text = extract_text_from_image(filename)
#     elif filename.lower().endswith('.pdf'):
#         extracted_text = extract_text_from_pdf(filename)
#     else:
#         print("Unsupported file format. Please upload a PNG, JPG, or PDF file.")
#         continue

#     # Extract the relevant transaction data
#     transactions_json = extract_transaction_data(extracted_text)

#     # Print the extracted data in JSON format
#     if (transactions_json["transactions"]["Deposits and Additions"] is not None or
#         transactions_json["transactions"]["ATM & DEBIT CARD WITHDRAWALS"] is not None or
#         transactions_json["transactions"]["ELECTRONIC WITHDRAWALS"] is not None or
#         transactions_json["transactions"]["OTHER WITHDRAWALS"] is not None):
#         print("\nExtracted Transactions (JSON Format):")
#         print(json.dumps(transactions_json, indent=4))
#     else:
#         print("No transactions found. Please check the OCR output and the regex pattern.")






import pytesseract
import cv2
import re
import fitz  # PyMuPDF for handling PDFs
import os
import json
from pdf2image import convert_from_path
import streamlit as st

# ✅ Extract text from an image (PNG, JPG, JPEG)
def extract_text_from_image(image_path):
    """Extracts text from an image file using OCR."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Unable to read image at {image_path}")
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        print(f"❌ Error during OCR: {e}")
        return ""

# ✅ Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using OCR."""
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            temp_image_path = f"page_{i}.jpg"
            image.save(temp_image_path, "JPEG")
            text += extract_text_from_image(temp_image_path)
            os.remove(temp_image_path)  # Clean up temp files
        return text
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return ""

# ✅ Extract balances from OCR text
def extract_balances(text):
    """Extracts balances dynamically from OCR text, handling multiple formats."""
    balance_patterns = {
        "beginning_balance": r"(?:Beginning|Opening|Prior) Balance[:\s]*\$?([\d,]+\.\d{2})",
        "income_total": r"Total Income[:\s]*\$?([\d,]+\.\d{2})",
        "expense_total": r"Total Expenses[:\s]*\$?([\d,]+\.\d{2})",
        "ending_balance": r"(?:Ending|Closing|Available) Balance[:\s]*\$?([\d,]+\.\d{2})"
    }

    extracted_balances = {
        "beginning_balance": "$0.00",
        "income_total": "$0.00",
        "expense_total": "$0.00",
        "ending_balance": "$0.00"
    }

    for key, pattern in balance_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_balances[key] = f"${match.group(1)}"

    return extracted_balances

# ✅ Extract transactions from OCR text
def extract_transaction_data(text):
    """Extract and categorize transactions dynamically from OCR text."""
    
    print("\n📌 Raw OCR Text Extracted:")
    print(text)

    # ✅ Extract balances dynamically from OCR text
    balances = extract_balances(text)

    # ✅ Extract transaction data dynamically
    pattern = r'(\d{2}/\d{2})\s+(.*?)\s+\$?([\d,]+\.\d{2})'
    matches = re.findall(pattern, text)

    deposits = []
    atm_withdrawals = []
    electronic_withdrawals = []
    other_withdrawals = []

    for date, description, amount in matches:
        # ✅ Fix: Clean up description but **keep internal numbers**
        cleaned_description = re.sub(r'^\d{2}/\d{2}', '', description)  # Remove leading date
        cleaned_description = re.sub(r'\s+\$?[\d,]+\.\d{2}$', '', cleaned_description)  # Remove trailing amount
        cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()  # Remove extra spaces

        transaction = {
            "date": date,
            "details": {
                "company_name": cleaned_description if cleaned_description else "",
                "origin_id": "",
                "date_description": "CO Entry",
                "name": cleaned_description.split()[0] if cleaned_description else "",
                "id": ""
            },
            "amount": f"${amount}"
        }

        # ✅ Improved Categorization Logic
        lower_desc = cleaned_description.lower()
        if "deposit" in lower_desc or "credit" in lower_desc:
            deposits.append(transaction)
        elif "atm" in lower_desc or "cash withdrawal" in lower_desc:
            atm_withdrawals.append(transaction)
        elif "transfer" in lower_desc or "electronic" in lower_desc or "zelle" in lower_desc:
            electronic_withdrawals.append(transaction)
        else:
            other_withdrawals.append(transaction)

    # ✅ Compute total income & expenses dynamically
    total_income = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in deposits) if deposits else 0.00
    total_expense = sum(float(t["amount"].replace("$", "").replace(",", "")) for t in atm_withdrawals + electronic_withdrawals + other_withdrawals) if (atm_withdrawals + electronic_withdrawals + other_withdrawals) else 0.00

    balances["income_total"] = f"${total_income:,.2f}"
    balances["expense_total"] = f"${total_expense:,.2f}"

    # ✅ Structured JSON output
    transactions_json = {
        "transactions": {
            "Deposits and Additions": deposits if deposits else None,
            "ATM Withdrawals": atm_withdrawals if atm_withdrawals else None,
            "Electronic Withdrawals": electronic_withdrawals if electronic_withdrawals else None,
            "Other Withdrawals": other_withdrawals if other_withdrawals else None
        },
        "balances": balances
    }

    return transactions_json

# ✅ Process uploaded file (PDF or Image)
def process_bank_statement(file_path):
    """Extract transactions from a bank statement (PDF or image)."""
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = extract_text_from_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_path)
    else:
        print("❌ Unsupported file format. Please upload a PNG, JPG, or PDF file.")
        return

    # ✅ Extract transactions from OCR text
    transactions_json = extract_transaction_data(extracted_text)

    # ✅ Save to JSON file
    output_folder = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output"
    os.makedirs(output_folder, exist_ok=True)  # ✅ Creates folder if it doesn't exist

    output_file = os.path.join(output_folder, "extracted_transactions.json")

    with open(output_file, "w") as json_file:
        json.dump(transactions_json, json_file, indent=4)

    print(f"\n✅ Transactions extracted and saved to `{output_file}`")
