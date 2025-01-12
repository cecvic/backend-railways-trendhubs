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
from typing import Dict, Any, Optional
from datetime import datetime

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS - This must be set up before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,  # Must be False for wildcard origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AnalysisRequest(BaseModel):
    stock_symbol: str
    analysis_type: str = "technical"

class PriceRange(BaseModel):
    high: float
    low: float

class Metrics(BaseModel):
    latest_close: float
    price_range: PriceRange
    volume: int

class Analysis(BaseModel):
    trend: str
    recommendation: str
    summary: str

class StockAnalysisResponse(BaseModel):
    stock_symbol: str
    metrics: Metrics
    analysis: Analysis
    timestamp: str

def extract_metrics_from_text(text: str) -> Dict[str, Any]:
    # Extract latest close price
    close_match = re.search(r'Latest close price: \$?([\d,.]+)', text)
    latest_close = float(close_match.group(1).replace(',', '')) if close_match else 0.0

    # Extract high price
    high_match = re.search(r'Highest price: \$?([\d,.]+)', text)
    high = float(high_match.group(1).replace(',', '')) if high_match else 0.0

    # Extract low price
    low_match = re.search(r'Lowest price: \$?([\d,.]+)', text)
    low = float(low_match.group(1).replace(',', '')) if low_match else 0.0

    # Extract volume
    volume_match = re.search(r'volume: (?:Approximately )?([\d,.]+)\s*(?:million)?', text, re.IGNORECASE)
    if volume_match:
        volume_str = volume_match.group(1).replace(',', '')
        volume = int(float(volume_str) * 1_000_000)  # Convert millions to actual number
    else:
        volume = 0

    return {
        "metrics": {
            "latest_close": latest_close,
            "price_range": {
                "high": high,
                "low": low
            },
            "volume": volume
        },
        "analysis": {
            "trend": extract_trend(text),
            "recommendation": extract_recommendation(text),
            "summary": text.strip()
        }
    }

def extract_trend(text: str) -> str:
    # Look for sentences containing trend information
    trend_patterns = [
        r'(?:has shown|showing|demonstrates|exhibited|displays|indicates).*(?:trend|movement|pattern)[^.]*\.',
        r'(?:upward|downward|bullish|bearish|neutral|sideways)[^.]*\.'
    ]
    
    for pattern in trend_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # If no specific trend sentence found, return first relevant sentence
    first_sentence = text.split('.')[0]
    return first_sentence + '.'

def extract_recommendation(text: str) -> str:
    # Look for recommendation patterns
    rec_patterns = [
        r'(?:recommended|recommend|suggest|advise).*?(?:\.|\n)',
        r'(?:could|should|might|may).*?(?:buy|sell|hold|watch|monitor).*?(?:\.|\n)',
        r'(?:good|great|excellent|perfect|ideal).*?(?:time|opportunity|moment).*?(?:to|for).*?(?:buy|sell|hold|watch|monitor).*?(?:\.|\n)'
    ]
    
    for pattern in rec_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # If no specific recommendation found, look for any sentence with trading-related terms
    trading_sentence = re.search(r'[^.]*(?:buy|sell|hold|watch|monitor)[^.]*\.', text, re.IGNORECASE)
    if trading_sentence:
        return trading_sentence.group(0).strip()
    
    return "Monitor the stock for further signals."

def get_analysis_function() -> dict:
    return {
        "name": "generate_stock_analysis",
        "description": "Generate a structured stock analysis report",
        "parameters": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "properties": {
                        "latest_close": {
                            "type": "number",
                            "description": "Latest closing price of the stock"
                        },
                        "price_range": {
                            "type": "object",
                            "properties": {
                                "high": {"type": "number"},
                                "low": {"type": "number"}
                            },
                            "required": ["high", "low"]
                        },
                        "volume": {
                            "type": "integer",
                            "description": "Trading volume"
                        }
                    },
                    "required": ["latest_close", "price_range", "volume"]
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "trend": {
                            "type": "string",
                            "description": "Current market trend analysis"
                        },
                        "recommendation": {
                            "type": "string",
                            "description": "Trading recommendation"
                        },
                        "summary": {
                            "type": "string",
                            "description": "Detailed analysis summary"
                        }
                    },
                    "required": ["trend", "recommendation", "summary"]
                }
            },
            "required": ["metrics", "analysis"]
        }
    }

def get_analysis_prompt(stock_symbol: str, analysis_type: str) -> str:
    prompts = {
        "technical": (
            f"You are a professional stock analyst. Analyze {stock_symbol} stock using technical analysis. "
            "Use real-time data to provide accurate metrics including latest close price, price range (high/low), "
            "and trading volume. Analyze price movements and technical indicators. "
            "You MUST include the following in your response:\n"
            "1. Latest close price with $ symbol\n"
            "2. Highest price with $ symbol\n"
            "3. Lowest price with $ symbol\n"
            "4. Trading volume\n"
            "5. Clear trend analysis\n"
            "6. Specific trading recommendations\n"
            "Format numbers clearly with $ for prices and 'million' for volume."
        ),
        "fundamental": (
            f"You are a professional stock analyst. Analyze {stock_symbol} stock using fundamental analysis. "
            "Use real-time data to provide accurate metrics including latest close price, price range (high/low), "
            "and trading volume. Analyze financial metrics and company performance. "
            "You MUST include the following in your response:\n"
            "1. Latest close price with $ symbol\n"
            "2. Highest price with $ symbol\n"
            "3. Lowest price with $ symbol\n"
            "4. Trading volume\n"
            "5. Clear market position analysis\n"
            "6. Specific investment recommendations\n"
            "Format numbers clearly with $ for prices and 'million' for volume."
        ),
        "sentiment": (
            f"You are a professional stock analyst. Analyze {stock_symbol} stock using sentiment analysis. "
            "Use real-time data to provide accurate metrics including latest close price, price range (high/low), "
            "and trading volume. Analyze market perception and news sentiment. "
            "You MUST include the following in your response:\n"
            "1. Latest close price with $ symbol\n"
            "2. Highest price with $ symbol\n"
            "3. Lowest price with $ symbol\n"
            "4. Trading volume\n"
            "5. Clear sentiment analysis\n"
            "6. Specific trading recommendations\n"
            "Format numbers clearly with $ for prices and 'million' for volume."
        )
    }
    return prompts.get(analysis_type, prompts["technical"])

def create_finance_assistant():
    return Assistant(
        llm=OpenAIChat(
            model="gpt-4",
            temperature=0.7,
        ),
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

@app.post("/analyze-stock", response_model=StockAnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    try:
        assistant = create_finance_assistant()
        
        # Get the analysis prompt and function definition
        prompt = get_analysis_prompt(request.stock_symbol, request.analysis_type)
        function_def = get_analysis_function()
        
        # Generate the analysis
        response = assistant.chat(
            prompt,
            functions=[function_def],
        )
        
        # Parse the response
        try:
            # Extract the function call result
            function_response = None
            response_text = ""
            
            # Collect all response chunks
            for chunk in response:
                if isinstance(chunk, dict):
                    if "function_call" in chunk:
                        try:
                            function_response = json.loads(chunk["function_call"]["arguments"])
                            break
                        except json.JSONDecodeError:
                            continue
                elif isinstance(chunk, str):
                    response_text += chunk
            
            # If we don't have a function response but have text, try to parse the text
            if not function_response and response_text:
                print("Attempting to parse text response:", response_text)
                function_response = extract_metrics_from_text(response_text)
            
            if not function_response:
                raise ValueError("No response data could be extracted")
            
            # Create the response object
            analysis_response = StockAnalysisResponse(
                stock_symbol=request.stock_symbol,
                metrics=Metrics(
                    latest_close=function_response["metrics"]["latest_close"],
                    price_range=PriceRange(
                        high=function_response["metrics"]["price_range"]["high"],
                        low=function_response["metrics"]["price_range"]["low"]
                    ),
                    volume=function_response["metrics"]["volume"]
                ),
                analysis=Analysis(
                    trend=function_response["analysis"]["trend"],
                    recommendation=function_response["analysis"]["recommendation"],
                    summary=function_response["analysis"]["summary"]
                ),
                timestamp=datetime.now().isoformat()
            )
            
            return analysis_response
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse analysis response: {str(e)}"
            )
            
    except Exception as e:
        print(f"Error in analyze_stock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}