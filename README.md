# 📄 ATS Resume Checker

A Streamlit app that scores your resume for Applicant Tracking Systems (ATS) and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Overall **ATS score (0-100)** with a weighted breakdown: keywords, formatting, sections, impact, readability
- Strengths, weaknesses and **missing keywords**
- Prioritised improvements (High / Medium / Low) and example bullet rewrites
- Optional **job description** box for better keyword matching
- Download the report as JSON

## Project structure
```
app.py            # Streamlit app
requirements.txt  # Python dependencies
README.md         # This file
```

## Run locally
1. Install Python 3.10 or newer.
2. Get a free Gemini API key: https://aistudio.google.com/apikey
3. Install and run:
```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
4. Paste your API key in the sidebar, upload a resume and click **Analyze resume**.

Optional: set the key once instead of typing it.
- Create `.streamlit/secrets.toml` with:
```toml
GEMINI_API_KEY = "your-key-here"
```
- Or set an environment variable `GEMINI_API_KEY`.

To change the default model, set the `GEMINI_MODEL` environment variable (default: `gemini-3.7-flash`). You can also pick a model in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this project to a **GitHub** repository (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** > deploy from an existing repo.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings > Secrets** and add:
```toml
GEMINI_API_KEY = "your-key-here"
```
6. Click **Deploy**. Your app gets a public URL after a minute or two.

## Push to GitHub
```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: ATS resume checker"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Create the empty repository on github.com first (no README/.gitignore added there). **Never commit your API key or `.streamlit/secrets.toml`.**

## Notes and limitations
- Scanned/image-only resumes have no text to read; export a text-based PDF or DOCX.
- The score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resume text is sent to the Gemini API for analysis; don't upload data you aren't comfortable sharing.

## Troubleshooting
| Problem | Fix |
|---|---|
| "503 UNAVAILABLE / high demand" | Google's servers are busy. The app retries and switches models automatically; if it still fails, wait a minute and retry |
| "Analysis failed ... model not found" | Pick another model in the sidebar or set `GEMINI_MODEL` |
| "Analysis failed ... API key" | Check the key; make sure it has no extra spaces |
| "Very little text was found" | Resume is likely a scanned image; use a text-based file |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
