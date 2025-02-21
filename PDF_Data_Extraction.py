# import os
# import cv2
# import pytesseract
# import re
# import fitz
# from pdf2image import convert_from_path
# import json
# from decimal import Decimal

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure correct path

# class TransactionProcessor:
#     # Define valid companies with their exact names
#     VALID_COMPANIES = {
#         "EPOS": ["Epos Now LLC"],
#         "DOORDASH": ["Doordash, Inc."],
#         "GRUBHUB": ["Grubhub Inc"],
#         "UBER": ["Uber USA", "Uber USA 6787"],
#         "INTUIT": ["Intuit Payroll S"],
#         "IRS": ["IRS"],
#         "WEBFILE": ["Webfile Tax Pymt"],
#         "SUPREME": ["Supreme Performa"],
#         "BEYOND": ["Beyond Menu"],
#         "SYSCO": ["Sysco Corporation"],
#         "AMTRUST": ["Amtrust NA"],
#         "ATM": ["ATM Withdrawal", "Card Purchase"]
#     }

#     @staticmethod
#     def clean_amount(amount: str) -> Decimal:
#         """Convert string amount to Decimal, handling currency symbols and commas."""
#         cleaned = amount.replace('$', '').replace(',', '')
#         return Decimal(cleaned)

#     @staticmethod
#     def format_amount(amount: Decimal) -> str:
#         """Format Decimal amount to currency string."""
#         return f"${amount:,.2f}"

#     def clean_company_name(self, description: str) -> str:
#         """Clean and extract company name from description."""
#         # Remove common prefixes
#         prefixes_to_remove = [
#             r"Orig CO Name:",
#             r"Orig ID:",
#             r"Desc Date:",
#             r"CO Entry",
#             r"\d{2}/\d{2}",
#             r"\$[\d,]+\.\d{2}",
#         ]
        
#         cleaned = description
#         for prefix in prefixes_to_remove:
#             cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
#         # Remove everything after certain markers
#         markers = ["Orig |D:", "Desc Date:", "Card 0611"]
#         for marker in markers:
#             if marker in cleaned:
#                 cleaned = cleaned.split(marker)[0]
        
#         # Clean up extra spaces and trim
#         cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
#         # Match with known company names
#         for company_type, variations in self.VALID_COMPANIES.items():
#             for variation in variations:
#                 if variation.lower() in cleaned.lower():
#                     return variation
                
#         # If no match found and it's a card purchase or ATM withdrawal
#         if "Card Purchase" in description or "ATM Withdrawal" in description:
#             return description.split(" ", 2)[0] + " " + description.split(" ", 2)[1]
            
#         # If no match found, return empty string
#         return ""

#     def categorize_transaction(self, description: str, amount: str) -> tuple[str, dict]:
#         """Categorize transaction and standardize company details."""
#         desc_upper = description.upper()
#         amount_dec = self.clean_amount(amount)
        
#         # Clean company name
#         company_name = self.clean_company_name(description)
        
#         # Get short name from company mapping
#         short_name = ""
#         for company_key, variations in self.VALID_COMPANIES.items():
#             if any(var.lower() in company_name.lower() for var in variations):
#                 short_name = company_key.capitalize()
#                 break

#         # Initialize transaction details
#         details = {
#             "company_name": company_name,
#             "origin_id": "",
#             "date_description": "CO Entry",
#             "name": short_name,
#             "id": ""
#         }

#         # Determine category
#         if "ATM WITHDRAWAL" in desc_upper or "CARD PURCHASE" in desc_upper:
#             return "ATM & DEBIT CARD WITHDRAWALS", details
#         elif any(keyword in desc_upper for keyword in ["INTUIT", "IRS", "PAYMENT", "SUPREME", "WEBFILE", "ZELLE"]):
#             return "ELECTRONIC WITHDRAWALS", details
#         elif "WITHDRAWAL" in desc_upper and not "ATM" in desc_upper:
#             return "OTHER WITHDRAWALS", details
#         else:
#             return "Deposits and Additions", details

#     def process_transactions(self, text: str) -> dict:
#         """Process and categorize transactions from text."""
#         # Initialize categories
#         categories = {
#             "Deposits and Additions": [],
#             "ATM & DEBIT CARD WITHDRAWALS": [],
#             "ELECTRONIC WITHDRAWALS": [],
#             "OTHER WITHDRAWALS": []
#         }
        
#         # Enhanced pattern to match transaction data
#         pattern = r'(\d{2}/\d{2})\s+(.*?)\s+\$?([\d,]+\.\d{2})'
#         matches = re.findall(pattern, text)

#         for date, description, amount in matches:
#             # Skip entries that look like balance summaries
#             if re.match(r'\$?[\d,]+\.\d{2}\s+\d{2}/\d{2}', description):
#                 continue
                
#             # Categorize transaction
#             category, details = self.categorize_transaction(description, amount)
            
#             # Only add transaction if we have a valid company name or it's a withdrawal
#             if details["company_name"] or "WITHDRAWAL" in category:
#                 transaction = {
#                     "date": date,
#                     "details": details,
#                     "amount": f"${amount}"
#                 }
#                 categories[category].append(transaction)

#         # Calculate balances
#         deposits = sum(self.clean_amount(t["amount"]) for t in categories["Deposits and Additions"])
#         withdrawals = sum(
#             self.clean_amount(t["amount"]) 
#             for cat in ["ATM & DEBIT CARD WITHDRAWALS", "ELECTRONIC WITHDRAWALS", "OTHER WITHDRAWALS"]
#             for t in categories[cat]
#         )

#         # Construct final JSON
#         return {
#             "transactions": {
#                 k: v if v else None for k, v in categories.items()
#             },
#             "check_summary": None,
#             "balances": {
#                 "beginning_balance": "$3,102.64", 
#                 "income_total": self.format_amount(deposits),
#                 "expense_total": self.format_amount(withdrawals),
#                 "ending_balance": self.format_amount(Decimal('3102.64') + deposits - withdrawals)
#             }
#         }

#     def process_file(self, file_path: str) -> dict:
#         """Process a file, extract transactions, and save to a JSON file."""
        
#         # Process the file based on type (PDF or Image)
#         if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image = cv2.imread(file_path)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             text = pytesseract.image_to_string(gray)
#         elif file_path.lower().endswith('.pdf'):
#             images = convert_from_path(file_path)
#             text = ""
#             for i, image in enumerate(images):
#                 temp_path = f"page_{i}.jpg"
#                 image.save(temp_path, "JPEG")
#                 text += pytesseract.image_to_string(cv2.imread(temp_path))
#                 os.remove(temp_path)
#         else:
#             raise ValueError("Unsupported file format")

#         # Process extracted text into structured transactions
#         transaction_data = self.process_transactions(text)

#         #  Save the extracted data to a JSON file
#         output_folder = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output"
#         output_file = os.path.join(output_folder, "extracted_transactions.json")

#         try:
#             os.makedirs(output_folder, exist_ok=True) 
#             with open(output_file, "w", encoding="utf-8") as json_file:
#                 json.dump(transaction_data, json_file, indent=4)
#             print(f"✅ Transactions saved successfully to {output_file}") 
#         except Exception as e:
#             print(f"❌ Error saving file: {e}")

#         return transaction_data  

import os
import cv2
import pytesseract
import re
from pdf2image import convert_from_path
import json
from decimal import Decimal
from flask import Flask, request, jsonify

# Set path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure correct path

class TransactionProcessor:
    # Define valid companies with their exact names
    VALID_COMPANIES = {
        "EPOS": ["Epos Now LLC"],
        "DOORDASH": ["Doordash, Inc."],
        "GRUBHUB": ["Grubhub Inc"],
        "UBER": ["Uber USA", "Uber USA 6787"],
        "INTUIT": ["Intuit Payroll S"],
        "IRS": ["IRS"],
        "WEBFILE": ["Webfile Tax Pymt"],
        "SUPREME": ["Supreme Performa"],
        "BEYOND": ["Beyond Menu"],
        "SYSCO": ["Sysco Corporation"],
        "AMTRUST": ["Amtrust NA"],
        "ATM": ["ATM Withdrawal", "Card Purchase"]
    }

    @staticmethod
    def clean_amount(amount: str) -> Decimal:
        """Convert string amount to Decimal, handling currency symbols and commas."""
        cleaned = amount.replace('$', '').replace(',', '')
        return Decimal(cleaned)

    @staticmethod
    def format_amount(amount: Decimal) -> str:
        """Format Decimal amount to currency string."""
        return f"${amount:,.2f}"

    def clean_company_name(self, description: str) -> str:
        """Clean and extract company name from description."""
        # Remove common prefixes
        prefixes_to_remove = [
            r"Orig CO Name:",
            r"Orig ID:",
            r"Desc Date:",
            r"CO Entry",
            r"\d{2}/\d{2}",
            r"\$[\d,]+\.\d{2}",
        ]
        
        cleaned = description
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        # Remove everything after certain markers
        markers = ["Orig |D:", "Desc Date:", "Card 0611"]
        for marker in markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]
        
        # Clean up extra spaces and trim
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Match with known company names
        for company_type, variations in self.VALID_COMPANIES.items():
            for variation in variations:
                if variation.lower() in cleaned.lower():
                    return variation
                
        # If no match found and it's a card purchase or ATM withdrawal
        if "Card Purchase" in description or "ATM Withdrawal" in description:
            parts = description.split(" ", 2)
            if len(parts) >= 2:
                return parts[0] + " " + parts[1]
            
        # If no match found, return empty string
        return ""

    def categorize_transaction(self, description: str, amount: str) -> tuple:
        """Categorize transaction and standardize company details."""
        desc_upper = description.upper()
        # Clean company name
        company_name = self.clean_company_name(description)
        
        # Get short name from company mapping
        short_name = ""
        for company_key, variations in self.VALID_COMPANIES.items():
            if any(var.lower() in company_name.lower() for var in variations):
                short_name = company_key.capitalize()
                break

        # Initialize transaction details
        details = {
            "company_name": company_name,
            "origin_id": "",
            "date_description": "CO Entry",
            "name": short_name,
            "id": ""
        }

        # Determine category
        if "ATM WITHDRAWAL" in desc_upper or "CARD PURCHASE" in desc_upper:
            return "ATM & DEBIT CARD WITHDRAWALS", details
        elif any(keyword in desc_upper for keyword in ["INTUIT", "IRS", "PAYMENT", "SUPREME", "WEBFILE", "ZELLE"]):
            return "ELECTRONIC WITHDRAWALS", details
        elif "WITHDRAWAL" in desc_upper and "ATM" not in desc_upper:
            return "OTHER WITHDRAWALS", details
        else:
            return "Deposits and Additions", details

    def extract_beginning_balance(self, text: str) -> str:
        """Extract beginning balance from the statement text."""
        patterns = [
            r"[Bb]eginning [Bb]alance.*?\$?([\d,]+\.\d{2})",
            r"[Pp]revious [Bb]alance.*?\$?([\d,]+\.\d{2})",
            r"[Oo]pening [Bb]alance.*?\$?([\d,]+\.\d{2})",
            r"[Bb]alance as of.*?\$?([\d,]+\.\d{2})",
            r"[Bb]alance on.*?\$?([\d,]+\.\d{2})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount = match.group(1)
                return f"${amount}"
        
        # Default to $0.00 if no pattern matches
        return "$0.00"

    def process_transactions(self, text: str) -> dict:
        """Process and categorize transactions from text."""
        beginning_balance = self.extract_beginning_balance(text)
        
        # Initialize categories
        categories = {
            "Deposits and Additions": [],
            "ATM & DEBIT CARD WITHDRAWALS": [],
            "ELECTRONIC WITHDRAWALS": [],
            "OTHER WITHDRAWALS": []
        }
        
        pattern = r'(\d{2}/\d{2})\s+(.*?)\s+\$?([\d,]+\.\d{2})'
        matches = re.findall(pattern, text)

        for date, description, amount in matches:
            if re.match(r'\$?[\d,]+\.\d{2}\s+\d{2}/\d{2}', description):
                continue
                
            category, details = self.categorize_transaction(description, amount)
            
            if details["company_name"] or "WITHDRAWAL" in category:
                transaction = {
                    "date": date,
                    "details": details,
                    "amount": f"${amount}"
                }
                categories[category].append(transaction)

        deposits = sum(self.clean_amount(t["amount"]) for t in categories["Deposits and Additions"])
        withdrawals = sum(
            self.clean_amount(t["amount"]) 
            for cat in ["ATM & DEBIT CARD WITHDRAWALS", "ELECTRONIC WITHDRAWALS", "OTHER WITHDRAWALS"]
            for t in categories[cat]
        )
        
        beginning_balance_decimal = self.clean_amount(beginning_balance)
        ending_balance = beginning_balance_decimal + deposits - withdrawals

        return {
            "transactions": {
                k: v if v else None for k, v in categories.items()
            },
            "check_summary": None,
            "balances": {
                "beginning_balance": beginning_balance,
                "income_total": self.format_amount(deposits),
                "expense_total": self.format_amount(withdrawals),
                "ending_balance": self.format_amount(ending_balance)
            }
        }

    def process_file(self, file_path: str) -> dict:
        """Process a file and return transaction data."""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
        elif file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            text = ""
            for i, image in enumerate(images):
                temp_path = f"page_{i}.jpg"
                image.save(temp_path, "JPEG")
                text += pytesseract.image_to_string(cv2.imread(temp_path))
                os.remove(temp_path)
        else:
            raise ValueError("Unsupported file format")
            
        # Process transactions
        result = self.process_transactions(text)
        
        output_folder = r"C:\Users\Admin\Desktop\Banking_chatbot\bankingproject\json_output"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "extracted_transaction.json")
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=4)
        print(f"✅ Transactions saved successfully to {output_file}")
        
        return result
