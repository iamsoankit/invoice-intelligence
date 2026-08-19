import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the modern Gemini client using your existing API key
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# --- Mock Data ---
mock_invoice_text = """
INVOICE #9942
Vendor: Global Aviation Supplies Ltd.
Total Amount: $45,000.00
Date: August 19, 2026
Terms: Due on Receipt. 
Note: Missing standard corporate tax ID on file.
"""

# --- Agent 1: Data Extractor ---
extractor_prompt = f"""
You are a Financial Data Extraction Specialist. 
Extract the Vendor Name, Total Amount, Date, and Terms from the following text. 
Output ONLY a valid JSON object without markdown formatting blocks.

Text to extract:
{mock_invoice_text}
"""

if __name__ == "__main__":
    print("--- Starting Agent 1: Data Extraction ---")
    
    extracted_response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=extractor_prompt
    )
    extracted_json = extracted_response.text.strip()
    print(extracted_json)
    
    # --- Agent 2: Risk Analyst ---
    analyst_prompt = f"""
    You are a meticulous corporate Risk Analyst. 
    Review the following JSON data. Flag any financial risks in a short bulleted list, 
    such as aggressive payment terms (e.g., Due on Receipt) or missing documentation. 
    If none, state 'No risks identified'.

    Extracted JSON Data:
    {extracted_json}
    """
    
    print("\n--- Starting Agent 2: Risk Analysis ---")
    
    risk_analysis = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=analyst_prompt
    )
    print(risk_analysis.text.strip())
