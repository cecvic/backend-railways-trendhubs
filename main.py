from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.assistant import Assistant
from phi.tools.yfinance import YFinanceTools
from phi.llm.openai import OpenAIChat
import json
import os
import re
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class AnalysisRequest(BaseModel):
    stock_symbol: str
    analysis_type: str = "technical"

class StockMetrics(BaseModel):
    latest_close: float
    price_range: Dict[str, float]
    volume: int
    trend: str
    recommendation: str

def clean_and_structure_analysis(raw_response: str, stock_symbol: str) -> Dict[str, Any]:
    # Remove markdown symbols and clean up the text
    clean_text = re.sub(r'[#*_\-|]', '', raw_response)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    try:
        # Extract key metrics using regex
        latest_close = float(re.search(r'Latest Close Price:\s*\$?(\d+\.?\d*)', clean_text).group(1))
        high = float(re.search(r'High:\s*\$?(\d+\.?\d*)', clean_text).group(1))
        low = float(re.search(r'Low:\s*\$?(\d+\.?\d*)', clean_text).group(1))
        volume_match = re.search(r'Volume:\s*(\d+)', clean_text)
        volume = int(volume_match.group(1)) if volume_match else 0
        
        # Extract trend and recommendation
        trend_match = re.search(r'Trend:(.*?)(?=\n|$)', clean_text)
        trend = trend_match.group(1).strip() if trend_match else ""
        
        recommendation_match = re.search(r'Recommendation:(.*?)(?=\n|$)', clean_text, re.DOTALL)
        recommendation = recommendation_match.group(1).strip() if recommendation_match else ""
        
        # Structure the response
        structured_response = {
            "stock_symbol": stock_symbol,
            "metrics": {
                "latest_close": latest_close,
                "price_range": {
                    "high": high,
                    "low": low
                },
                "volume": volume,
            },
            "analysis": {
                "trend": trend,
                "recommendation": recommendation,
                "summary": clean_text.strip()
            },
            "timestamp": "",  # You might want to add the current timestamp here
        }
        
        return structured_response
        
    except (AttributeError, ValueError) as e:
        print(f"Error structuring analysis: {e}")
        return {
            "stock_symbol": stock_symbol,
            "analysis": clean_text.strip()
        }

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://*.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
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
            analysis_prompt = (
                f"Perform a detailed technical analysis for {request.stock_symbol} stock. "
                "Include the following metrics: Latest close price, price range (high/low), "
                "volume, clear trend analysis, and specific trading recommendations. "
                "Format the response in a clear, professional manner without using markdown symbols."
            )
        elif request.analysis_type == "fundamental":
            analysis_prompt = (
                f"Provide a comprehensive fundamental analysis for {request.stock_symbol} stock. "
                "Include key metrics, financial ratios, growth prospects, and market position. "
                "Format the response in a clear, professional manner without using markdown symbols."
            )
        elif request.analysis_type == "sentiment":
            analysis_prompt = (
                f"Analyze current market sentiment for {request.stock_symbol} stock. "
                "Include social media trends, news sentiment, and analyst opinions. "
                "Format the response in a clear, professional manner without using markdown symbols."
            )
        else:
            analysis_prompt = (
                f"Provide a professional analysis of {request.stock_symbol} stock. "
                "Include key metrics and clear recommendations. "
                "Format the response in a clear, professional manner without using markdown symbols."
            )
        
        response_generator = assistant.chat(analysis_prompt)
        response = collect_generator_response(response_generator)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate analysis")
            
        # Clean and structure the response
        structured_response = clean_and_structure_analysis(response.strip(), request.stock_symbol)
        return structured_response
            
    except Exception as e:
        print(f"Error in analyze_stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}