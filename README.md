# Invoice Intelligence ⚡

A real-time, multi-agent enterprise invoice intelligence tool built with Python, Streamlit, and the Google GenAI SDK. Audit unstructured vendor documents, evaluate risk tiers, and export clean data seamlessly!

## Features

*   **Multi-Agent Pipeline:** Extractor node parses raw, unstructured vendor documents into strict JSON schemas, and Auditor node evaluates corporate compliance rules.
*   **Risk Tiers:** Dynamic color-coded risk assessment (Critical, Moderate, Secure) with actionable security insights.
*   **Demo Mode:** Instant local simulation to test and showcase features seamlessly without hitting API rate limits.
*   **Data Export:** Export audited payloads instantly as structured JSON or CSV for ERP integration.
*   **Neo-Brutalist UI:** High-contrast, modern developer-focused custom styling built natively in Streamlit.

## Technologies Used

*   Python
*   Streamlit
*   Google Gemini API (`google-genai` SDK)
*   Python-Dotenv (Environment Security)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/iamsoankit/invoice-intelligence.git](https://github.com/iamsoankit/invoice-intelligence.git)
    cd invoice-intelligence
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure your environment:**
    Create a `.env` file in the root directory and add your API key:
    ```bash
    echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
    ```
4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application should now be running on `http://localhost:8501`.

## How to Use

*   **Input Mode:** Choose between uploading a sample invoice or entering raw vendor data text.
*   **Run Pipeline:** Trigger the multi-agent extraction and auditing nodes.
*   **Review & Export:** Inspect the parsed JSON payload, check the compliance risk tier badge, and download your report as JSON or CSV.

## Notes

*   Make sure your Gemini API key is properly added to your local `.env` file and **never** commit it to version control.
*   If you encounter any module errors, ensure your virtual environment is active before running `pip install` or starting the app.
