from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.assistant import Assistant
from phi.tools.yfinance import YFinanceTools
from phi.llm.openai import OpenAIChat
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AnalysisRequest(BaseModel):
    stock_symbol: str
    analysis_type: str = "technical"

app = FastAPI()

# Update CORS middleware to be more production-ready
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.netlify.app",  # Netlify domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_finance_assistant():
    return Assistant(
        llm=OpenAIChat(model="gpt-4o-mini"),
        tools=[YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True, 
            company_news=True, 
            key_financial_ratios=True, 
            income_statements=True, 
            technical_indicators=True
        )],
        show_tool_calls=True,
    )

def collect_generator_response(generator):
    response_chunks = []
    try:
        for chunk in generator:
            if chunk:
                response_chunks.append(str(chunk))
    except Exception as e:
        print(f"Error collecting response: {e}")
    return "".join(response_chunks)

@app.post("/analyze-stock")
async def analyze_stock(request: AnalysisRequest):
    try:
        assistant = create_finance_assistant()
        
        # Customize analysis based on type
        if request.analysis_type == "technical":
            analysis_prompt = f"Perform technical analysis on {request.stock_symbol} stock"
        elif request.analysis_type == "fundamental":
            analysis_prompt = f"Provide fundamental analysis for {request.stock_symbol} stock"
        elif request.analysis_type == "sentiment":
            analysis_prompt = f"Analyze market sentiment for {request.stock_symbol} stock"
        elif request.analysis_type == "comparative":
            analysis_prompt = f"Compare {request.stock_symbol} stock with its peers"
        elif request.analysis_type == "news_based":
            analysis_prompt = f"Analyze recent news impact on {request.stock_symbol} stock"
        elif request.analysis_type == "risk":
            analysis_prompt = f"Evaluate risks for {request.stock_symbol} stock"
        else:
            analysis_prompt = f"Analyze {request.stock_symbol} stock"
        
        response_generator = assistant.chat(analysis_prompt)
        response = collect_generator_response(response_generator)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate analysis")
            
        response = response.strip()
        
        try:
            parsed_response = json.loads(response)
            return {
                "stock_symbol": request.stock_symbol,
                "analysis": parsed_response
            }
        except json.JSONDecodeError:
            return {
                "stock_symbol": request.stock_symbol,
                "analysis": response
            }
            
    except Exception as e:
        print(f"Error in analyze_stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}