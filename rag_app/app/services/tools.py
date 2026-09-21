"""
External tools service for Calculator, Weather, and YouTube searches.
No API keys needed for YouTube (uses links directly).
"""
import json
import logging
import re
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base exception for tool errors."""

    pass


class CalculatorTool:
    """Simple calculator tool for math expressions."""

    @staticmethod
    def execute(expression: str) -> dict:
        """
        Safely evaluate math expressions.

        Args:
            expression: Math expression (e.g., "2 + 2", "10 * 5")

        Returns:
            dict with result or error
        """
        try:
            # Remove spaces and validate input
            expr = expression.strip()
            
            # Only allow safe characters
            if not re.match(r'^[\d+\-*/().\s%]+$', expr):
                return {"status": "error", "message": "Invalid expression. Only numbers and math operators allowed."}
            
            # Evaluate safely
            result = eval(expr)  # noqa: S307 - controlled input only
            return {
                "status": "success",
                "result": result,
                "expression": expr,
                "tool": "calculator",
            }
        except ZeroDivisionError:
            return {"status": "error", "message": "Division by zero error."}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"Calculation error: {str(e)}"}


class WeatherTool:
    """Weather tool using Open-Meteo (no API key required)."""

    API_URL = "https://api.open-meteo.com/v1/forecast"

    @staticmethod
    def get_coordinates(city: str) -> dict:
        """Get latitude/longitude from city name."""
        try:
            geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": city, "count": 1, "language": "en", "format": "json"}
            
            response = requests.get(geocoding_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if not data.get("results"):
                return {"status": "error", "message": f"City '{city}' not found."}
            
            result = data["results"][0]
            return {
                "status": "success",
                "name": result.get("name"),
                "country": result.get("country"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
            }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Geocoding error: {str(e)}"}

    @staticmethod
    def execute(city: str) -> dict:
        """
        Get weather for a city.

        Args:
            city: City name

        Returns:
            dict with weather info or error
        """
        # Get coordinates first
        coords = WeatherTool.get_coordinates(city)
        if coords.get("status") == "error":
            return coords

        try:
            params = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit",
            }
            
            response = requests.get(WeatherTool.API_URL, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            current = data.get("current", {})
            
            return {
                "status": "success",
                "city": coords["name"],
                "country": coords["country"],
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "tool": "weather",
            }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Weather API error: {str(e)}"}


class YouTubeTool:
    """YouTube search tool (generates search URLs - no API key needed)."""

    SEARCH_URL = "https://www.youtube.com/results"

    @staticmethod
    def execute(query: str) -> dict:
        """
        Generate YouTube search link.

        Args:
            query: Search query

        Returns:
            dict with YouTube search URL
        """
        try:
            encoded_query = quote(query)
            search_url = f"{YouTubeTool.SEARCH_URL}?search_query={encoded_query}"
            
            return {
                "status": "success",
                "query": query,
                "search_url": search_url,
                "tool": "youtube",
                "message": f"Click the link below to watch videos about '{query}'",
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"YouTube search error: {str(e)}"}


class ToolExecutor:
    """Execute external tools based on user question."""

    TOOLS = {
        "calculator": CalculatorTool(),
        "weather": WeatherTool(),
        "youtube": YouTubeTool(),
    }

    # Keywords that trigger each tool
    TRIGGERS = {
        "calculator": r'\b(calculate|math|compute|what is|how much|solve|equation|\+|-|\*|/|equals)\b',
        "weather": r'\b(weather|temperature|forecast|rain|snow|sunny|cloudy|climate)\b',
        "youtube": r'\b(youtube|video|watch|how to|tutorial|show me|find videos?)\b',
    }

    @staticmethod
    def detect_tool(question: str) -> tuple[str | None, dict]:
        """
        Detect if external tool should be used.

        Args:
            question: User's question

        Returns:
            (tool_name, parsed_data) or (None, {})
        """
        question_lower = question.lower()

        # Check weather
        if re.search(ToolExecutor.TRIGGERS["weather"], question_lower):
            # Extract city name (simple heuristic)
            match = re.search(r'(?:in|at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', question)
            if match:
                city = match.group(1)
                return "weather", {"city": city}
            # Try to extract a proper noun
            words = question_lower.split()
            for i, word in enumerate(words):
                if word in ["in", "at", "for"] and i + 1 < len(words):
                    city = question.split()[i + 1]
                    return "weather", {"city": city}

        # Check calculator
        if re.search(ToolExecutor.TRIGGERS["calculator"], question_lower):
            # Extract math expression
            match = re.search(r'([\d+\-*/().\s%]+)', question)
            if match:
                expr = match.group(1).strip()
                if re.match(r'^[\d+\-*/().\s%]+$', expr):
                    return "calculator", {"expression": expr}

        # Check YouTube
        if re.search(ToolExecutor.TRIGGERS["youtube"], question_lower):
            return "youtube", {"query": question}

        return None, {}

    @staticmethod
    def execute(tool_name: str, params: dict) -> dict:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool result
        """
        if tool_name not in ToolExecutor.TOOLS:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        tool = ToolExecutor.TOOLS[tool_name]
        try:
            if tool_name == "calculator":
                return tool.execute(params.get("expression", ""))
            elif tool_name == "weather":
                return tool.execute(params.get("city", ""))
            elif tool_name == "youtube":
                return tool.execute(params.get("query", ""))
        except Exception as e:  # noqa: BLE001
            logger.error("Tool execution error for %s: %s", tool_name, e)
            return {"status": "error", "message": f"Tool execution failed: {str(e)}"}

        return {"status": "error", "message": "Unknown error"}



## mmr to foolow