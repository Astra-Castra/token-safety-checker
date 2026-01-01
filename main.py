"""Token Safety Checker - LangGraph Agent (Multi-Chain)
Supports: Ethereum, BSC, Polygon, Arbitrum, Optimism, Base, Solana, Tron, Sui, Hyperliquid, TON, opBNB
Can be deployed anywhere (Railway, Render, Fly.io, etc.)
NO LangGraph Cloud subscription needed!
"""

from __future__ import annotations

import os
import re
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# LangGraph State and Context
class Context(TypedDict):
    """Context parameters for the agent."""
    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent."""
    messages: List[Dict[str, Any]]


class TokenSafetyAnalyzer:
    """Token Safety Analysis Tools"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.goplus_api = "https://api.gopluslabs.io/api/v1/token_security"
        self.dexscreener_api = "https://api.dexscreener.com/latest/dex/tokens"
        self.openai_client = openai_client
        
        # Chain ID mapping for GoPlus API
        self.chain_map = {
            # EVM Chains
            "eth": "1",
            "ethereum": "1",
            "bsc": "56",
            "binance": "56",
            "polygon": "137",
            "matic": "137",
            "arbitrum": "42161",
            "optimism": "10",
            "base": "8453",
            "opbnb": "204",
            "avalanche": "43114",
            "avax": "43114",
            "fantom": "250",
            "ftm": "250",
            "cronos": "25",
            "cro": "25",
            # Non-EVM Chains
            "solana": "solana",
            "sol": "solana",
            "tron": "tron",
            "trx": "tron",
            "sui": "sui",
            "ton": "ton",
            "hyperliquid": "hyperliquid"
        }
    
    def extract_token_address(self, text: str) -> tuple:
        """Extract token address from text - supports EVM and non-EVM formats"""
        # Try EVM format (0x + 40 hex chars)
        evm_pattern = r'0x[a-fA-F0-9]{40}'
        evm_match = re.search(evm_pattern, text)
        if evm_match:
            return evm_match.group(0), "evm"
        
        # Try Solana format (base58, 32-44 chars)
        solana_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        solana_match = re.search(solana_pattern, text)
        if solana_match:
            return solana_match.group(0), "solana"
        
        # Try Tron format (T + base58)
        tron_pattern = r'\bT[1-9A-HJ-NP-Za-km-z]{33}\b'
        tron_match = re.search(tron_pattern, text)
        if tron_match:
            return tron_match.group(0), "tron"
        
        return None, None
    
    async def check_goplus_security(self, token_address: str, chain: str = "eth") -> dict:
        """Query GoPlus Labs API for security data"""
        chain_id = self.chain_map.get(chain.lower(), "1")
        
        # For non-EVM chains, GoPlus uses different endpoints
        if chain_id in ["solana", "tron", "sui", "ton", "hyperliquid"]:
            # Special handling for non-EVM chains
            if chain_id == "solana":
                url = "https://api.gopluslabs.io/api/v1/token_security/solana"
            elif chain_id == "tron":
                url = "https://api.gopluslabs.io/api/v1/token_security/tron"
            else:
                # For chains not yet fully supported by GoPlus
                return self._limited_analysis(token_address, chain)
        else:
            url = f"{self.goplus_api}/{chain_id}"
        
        params = {"contract_addresses": token_address}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        token_data = result.get(token_address.lower(), {})
                        
                        if not token_data:
                            # Try without lowercasing for non-EVM chains
                            token_data = result.get(token_address, {})
                        
                        return {
                            "is_honeypot": token_data.get("is_honeypot", "0") == "1",
                            "is_open_source": token_data.get("is_open_source", "0") == "1",
                            "is_mintable": token_data.get("is_mintable", "0") == "1",
                            "can_take_back_ownership": token_data.get("can_take_back_ownership", "0") == "1",
                            "owner_change_balance": token_data.get("owner_change_balance", "0") == "1",
                            "hidden_owner": token_data.get("hidden_owner", "0") == "1",
                            "selfdestruct": token_data.get("selfdestruct", "0") == "1",
                            "trading_cooldown": token_data.get("trading_cooldown", "0") == "1",
                            "buy_tax": float(token_data.get("buy_tax", "0")),
                            "sell_tax": float(token_data.get("sell_tax", "0")),
                            "holder_count": int(token_data.get("holder_count", "0")),
                        }
                    return self._limited_analysis(token_address, chain)
        except Exception as e:
            print(f"GoPlus API error: {str(e)}")
            return self._limited_analysis(token_address, chain)
    
    def _limited_analysis(self, token_address: str, chain: str) -> dict:
        """Return limited analysis when full API data unavailable"""
        return {
            "is_honeypot": False,
            "is_open_source": False,
            "is_mintable": False,
            "can_take_back_ownership": False,
            "owner_change_balance": False,
            "hidden_owner": False,
            "selfdestruct": False,
            "trading_cooldown": False,
            "buy_tax": 0.0,
            "sell_tax": 0.0,
            "holder_count": 0,
            "limited_data": True
        }
    
    async def check_dexscreener(self, token_address: str) -> dict:
        """Query DexScreener API for liquidity data - supports all chains"""
        url = f"{self.dexscreener_api}/{token_address}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])
                        
                        if not pairs:
                            return {}
                        
                        main_pair = max(pairs, key=lambda x: float(x.get("liquidity", {}).get("usd", 0)))
                        liquidity_data = main_pair.get("liquidity", {})
                        
                        return {
                            "liquidity_usd": float(liquidity_data.get("usd", 0)),
                            "price_usd": float(main_pair.get("priceUsd", 0)),
                            "volume_24h": float(main_pair.get("volume", {}).get("h24", 0)),
                            "dex_name": main_pair.get("dexId", "Unknown"),
                            "chain_id": main_pair.get("chainId", "unknown")
                        }
                    return {}
        except Exception as e:
            print(f"DexScreener API error: {str(e)}")
            return {}
    
    def calculate_risk_score(self, goplus_data: dict, dex_data: dict) -> tuple:
        """Calculate overall risk score (0-100, higher = riskier)"""
        risk_score = 0.0
        flags = []
        
        # Check if data is limited
        if goplus_data.get("limited_data"):
            flags.append("ℹ️ Limited security data available for this chain")
        
        if goplus_data.get("is_honeypot", False):
            risk_score += 30
            flags.append("🚨 HONEYPOT DETECTED")
        
        if goplus_data.get("hidden_owner", False):
            risk_score += 20
            flags.append("🚨 Hidden owner detected")
        
        if goplus_data.get("selfdestruct", False):
            risk_score += 20
            flags.append("🚨 Can self-destruct")
        
        if goplus_data.get("can_take_back_ownership", False):
            risk_score += 15
            flags.append("⚠️ Owner can take back ownership")
        
        if goplus_data.get("owner_change_balance", False):
            risk_score += 10
            flags.append("⚠️ Owner can change balances")
        
        if goplus_data.get("is_mintable", False):
            risk_score += 10
            flags.append("⚠️ Token is mintable")
        
        buy_tax = goplus_data.get("buy_tax", 0)
        sell_tax = goplus_data.get("sell_tax", 0)
        
        if buy_tax > 10 or sell_tax > 10:
            risk_score += 10
            flags.append(f"⚠️ High taxes: Buy {buy_tax}%, Sell {sell_tax}%")
        
        liquidity_usd = dex_data.get("liquidity_usd", 0)
        if liquidity_usd > 0 and liquidity_usd < 10000:
            risk_score += 15
            flags.append(f"⚠️ Low liquidity: ${liquidity_usd:,.2f}")
        
        if goplus_data.get("trading_cooldown", False):
            risk_score += 5
            flags.append("⚠️ Trading cooldown enabled")
        
        if not goplus_data.get("is_open_source", False):
            risk_score += 5
            flags.append("⚠️ Contract not verified")
        
        return min(risk_score, 100.0), flags
    
    async def analyze_token(self, token_address: str, chain: str = "eth") -> str:
        """Main analysis function"""
        goplus_data, dex_data = await asyncio.gather(
            self.check_goplus_security(token_address, chain),
            self.check_dexscreener(token_address)
        )
        
        if not goplus_data and not dex_data:
            return f"❌ Unable to fetch data for token {token_address} on {chain.upper()}"
        
        risk_score, flags = self.calculate_risk_score(goplus_data, dex_data)
        
        if risk_score >= 50:
            recommendation = "❌ AVOID - High Risk"
        elif risk_score >= 25:
            recommendation = "⚠️ PROCEED WITH CAUTION"
        else:
            recommendation = "✅ RELATIVELY SAFE"
        
        report = f"""🛡️ TOKEN SAFETY ANALYSIS

📍 Token: {token_address}
⛓️ Chain: {chain.upper()}

📊 RISK SCORE: {risk_score:.1f}/100
🎯 RECOMMENDATION: {recommendation}

"""
        
        if dex_data.get("liquidity_usd"):
            report += f"""💰 Liquidity: ${dex_data['liquidity_usd']:,.2f}
📈 24h Volume: ${dex_data.get('volume_24h', 0):,.2f}
💵 Price: ${dex_data.get('price_usd', 0):.8f}
🔄 DEX: {dex_data.get('dex_name', 'Unknown')}

"""
        
        if goplus_data.get("holder_count"):
            report += f"👥 Holders: {goplus_data['holder_count']:,}\n\n"
        
        if flags:
            report += f"🚩 RISK FLAGS:\n"
            for flag in flags:
                report += f"  • {flag}\n"
        else:
            report += "✅ No major risk flags detected\n"
        
        if goplus_data.get("buy_tax") or goplus_data.get("sell_tax"):
            report += f"\n📊 Taxes: Buy {goplus_data.get('buy_tax', 0)}% / Sell {goplus_data.get('sell_tax', 0)}%"
        
        return report


# Initialize OpenAI client
# Lazy-load OpenAI client to avoid startup errors
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return AsyncOpenAI(api_key=api_key)

def get_analyzer():
    return TokenSafetyAnalyzer(get_openai_client())


async def call_model(state: State) -> Dict[str, Any]:
    """Process conversational messages using LangGraph"""

    analyzer = get_analyzer()
    
    latest_message = state.messages[-1] if state.messages else {}
    user_content = latest_message.get("content", "")
    
    # Extract token address (supports multiple formats)
    token_address, address_type = analyzer.extract_token_address(user_content)
    
    # Determine chain from message
    chain = "eth"
    user_lower = user_content.lower()
    
    # Check for chain mentions
    if "bsc" in user_lower or "binance" in user_lower:
        chain = "bsc"
    elif "polygon" in user_lower or "matic" in user_lower:
        chain = "polygon"
    elif "arbitrum" in user_lower:
        chain = "arbitrum"
    elif "optimism" in user_lower:
        chain = "optimism"
    elif "base" in user_lower:
        chain = "base"
    elif "solana" in user_lower or "sol" in user_lower:
        chain = "solana"
    elif "tron" in user_lower or "trx" in user_lower:
        chain = "tron"
    elif "sui" in user_lower:
        chain = "sui"
    elif "ton" in user_lower:
        chain = "ton"
    elif "hyperliquid" in user_lower:
        chain = "hyperliquid"
    elif "opbnb" in user_lower:
        chain = "opbnb"
    
    # Analyze token if address found
    if token_address:
        try:
            ai_response = await analyzer.analyze_token(token_address, chain)
        except Exception as e:
            ai_response = f"❌ Error analyzing token: {str(e)}"
    else:
        # No token address found
        ai_response = """I'm a Token Safety Checker. Please provide a token contract address to analyze.

Supported chains:
• Ethereum, BSC, Polygon, Arbitrum, Optimism, Base, opBNB
• Solana, Tron, Sui, TON, Hyperliquid

Example: 0x2170ed0880ac9a755fd29b2688956bd959f933f8 on BSC"""
    
    response_message = {
        "role": "assistant",
        "content": ai_response
    }
    
    return {
        "messages": state.messages + [response_message]
    }


# Build LangGraph
graph = (
    StateGraph(State)
    .add_node("call_model", call_model)
    .add_edge("__start__", "call_model")
    .compile()
)


# FastAPI wrapper for HTTP access
app = FastAPI(title="Token Safety Checker API - Multi-Chain")

# ✅ ENABLE CORS - THIS FIXES THE BROWSER ERROR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class MessageRequest(BaseModel):
    content: str
    chain: str = "eth"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Token Safety Checker API - Multi-Chain Support",
        "supported_chains": [
            "Ethereum (eth)", "BSC (bsc)", "Polygon (polygon)", 
            "Arbitrum (arbitrum)", "Optimism (optimism)", "Base (base)",
            "Solana (solana)", "Tron (tron)", "Sui (sui)", 
            "TON (ton)", "Hyperliquid (hyperliquid)", "opBNB (opbnb)"
        ],
        "endpoints": {
            "/analyze": "POST - Analyze a token",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0-multi-chain"}


@app.post("/analyze")
async def analyze_endpoint(request: MessageRequest):
    """Analyze a token via HTTP POST"""
    try:
        # Create state for LangGraph
        state = State(messages=[{
            "role": "user",
            "content": f"{request.content} on {request.chain}"
        }])
        
        # Run the graph
        result = await graph.ainvoke(state)
        
        # Extract response
        response_content = result["messages"][-1]["content"]
        
        return JSONResponse({
            "success": True,
            "analysis": response_content
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)