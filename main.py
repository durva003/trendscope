from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os, json, requests
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AnalyzeRequest(BaseModel):
    product: str
    limit: int = 10

def scrape_mentions(product: str, limit: int = 10):
    headers = {"User-Agent": "Mozilla/5.0"}
    query = f"{product} review pros cons"
    url = f"https://html.duckduckgo.com/html/?q={query}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        results = soup.find_all("a", class_="result__snippet")
        return [r.get_text(strip=True) for r in results[:limit]]
    except Exception as e:
        return []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    mentions = scrape_mentions(req.product, req.limit)

    if not mentions:
        return {"error": "Could not fetch data. Try again."}

    combined = "\n".join(mentions)

    prompt = (
        f'You are a product manager analyzing user feedback for "{req.product}".\n\n'
        f'Here are real web mentions and reviews:\n{combined}\n\n'
        'Return ONLY a valid JSON object with this exact structure, no extra text:\n'
        '{\n'
        '  "sentiment_score": 7.5,\n'
        '  "summary": "two sentence summary here",\n'
        '  "top_praises": ["praise 1", "praise 2", "praise 3"],\n'
        '  "top_complaints": ["complaint 1", "complaint 2", "complaint 3"],\n'
        '  "feature_requests": ["request 1", "request 2", "request 3"],\n'
        '  "priority_fixes": ["fix 1", "fix 2", "fix 3"],\n'
        '  "persona": "one sentence describing typical user",\n'
        '  "verdict": "one punchy PM takeaway"\n'
        '}'
    )

    try:
        response = client.chat.completions.create(
            model= "llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw = response.choices[0].message.content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        data["reviews_analyzed"] = len(mentions)
        return data
    except json.JSONDecodeError:
        return {"error": "AI returned invalid JSON. Try again."}
    except Exception as e:
        return {"error": f"Groq error: {str(e)}"}