# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An estimated **ATS score** (0-100) with a category breakdown
- **Strengths, weaknesses and missing keywords**
- **Prioritised, specific improvements** (with rewrite suggestions)
- A downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

> The ATS score is an AI-generated estimate based on common ATS best practices.
> Real ATS software varies by employer and does not publish a standard score.

## How the score works

Gemini rates five categories from 0-100. The app then computes the final score itself, so it is consistent:

| Category | Weight |
|---|---|
| Formatting & structure | 20% |
| Keywords & skills | 25% |
| Experience & impact | 25% |
| Readability & language | 15% |
| Contact info & sections | 15% |

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration

- Default model is `gemini-2.5-flash`. You can change it in the sidebar, or edit `DEFAULT_MODEL` in `app.py`.
- Never commit your API key to GitHub. Use Streamlit secrets or an environment variable.

## Limitations

- Scanned or image-only PDFs can't be read (an ATS couldn't read them either).
- Max upload size is 5 MB.
- Resume content is sent to the Gemini API for analysis. Don't upload documents you aren't comfortable sharing.
