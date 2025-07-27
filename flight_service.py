import os
import re
import requests
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging

from iata_codes import get_iata_lookup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightService:
    """Service for handling flight information using AviationStack API."""
    
    BASE_URL = "http://api.aviationstack.com/v1"
    
    def __init__(self, api_key: str = None):
        """Initialize the FlightService with API key.
        
        Args:
            api_key: AviationStack API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY")
        if not self.api_key:
            logger.warning("No AviationStack API key provided. Flight data will not be available.")
            
        # Initialize IATA code lookup
        self.iata_lookup = get_iata_lookup(self.api_key)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the AviationStack API.
        
        Args:
            endpoint: API endpoint (e.g., 'flights')
            params: Query parameters
            
        Returns:
            Dict containing the API response or error information
        """
        if not self.api_key:
            return {
                "error": "No API key provided. "
                        "Please provide your AviationStack API key in the .env file as AVIATIONSTACK_API_KEY."
            }
            
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params.update({
            "access_key": self.api_key,
        })
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Check for API errors
            if response.status_code == 403:
                return {
                    "error": "API access denied. Please check if your API key is valid and has the required permissions."
                }
            elif response.status_code == 429:
                return {
                    "error": "API rate limit exceeded. Please try again later or upgrade your subscription."
                }
                
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Check for API-specific errors
            if isinstance(data, dict) and 'error' in data:
                return {"error": f"API Error: {data.get('error', {}).get('info', 'Unknown error')}"}
                
            return data
            
        except requests.exceptions.Timeout:
            return {"error": "The request to the flight information service timed out. Please try again later."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to AviationStack API: {e}")
            return {"error": f"Failed to fetch flight data: {str(e)}"}
    
    def get_flight_status(self, flight_number: str = None, flight_iata: str = None) -> Dict[str, Any]:
        """Get the status of a specific flight.
        
        Args:
            flight_number: Flight number (e.g., '123')
            flight_iata: IATA flight code (e.g., 'UL123')
            
        Returns:
            Dict containing flight status information
        """
        if not (flight_number or flight_iata):
            return {"error": "Either flight_number or flight_iata must be provided"}
            
        params = {"flight_iata": flight_iata} if flight_iata else {"flight_number": flight_number}
        return self._make_request("flights", params)
    
    def _parse_location_input(self, location: str) -> Tuple[Optional[str], str]:
        """Parse location input which could be an IATA code or a city/airport name.
        
        Args:
            location: IATA code or location name
            
        Returns:
            Tuple of (iata_code, display_name)
        """
        if not location:
            return None, ""
            
        # Check if it's already an IATA code (2-4 letters, all caps)
        if re.match(r'^[A-Z]{2,4}$', location.upper()):
            return location.upper(), location.upper()
            
        # Try to look up the IATA code
        airport_info = self.iata_lookup.get_airport_info(location)
        if airport_info:
            return airport_info['iata_code'], f"{airport_info['name']} ({airport_info['iata_code']})"
            
        # If not found, return the original input
        return None, location
    
    def get_flights_by_route(self, dep_location: str, arr_location: str, date: str = None) -> Dict[str, Any]:
        """Get flights between two locations.
        
        Args:
            dep_location: Departure location (IATA code, city, or airport name)
            arr_location: Arrival location (IATA code, city, or airport name)
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dict containing flight information
        """
        # Parse departure location
        dep_iata, dep_display = self._parse_location_input(dep_location)
        if not dep_iata:
            return {"error": f"Could not identify departure airport from: {dep_location}"}
            
        # Parse arrival location
        arr_iata, arr_display = self._parse_location_input(arr_location)
        if not arr_iata:
            return {"error": f"Could not identify arrival airport from: {arr_location}"}
            
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
            
        params = {
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
            "flight_date": date
        }
        
        # Add display names to the response
        result = self._make_request("flights", params)
        if not result.get('error'):
            result['departure'] = dep_display
            result['arrival'] = arr_display
            result['date'] = date
            
        return result
        
    def _get_airport_code(self, location: str) -> str:
        """Convert location name to IATA airport code.
        
        Args:
            location: City or airport name
            
        Returns:
            IATA airport code or original string if not found
        """
        # Simple mapping of common cities to their main airport codes
        # In a production app, this would be more comprehensive
        airport_codes = {
            'colombo': 'CMB',
            'cmb': 'CMB',
            'bandaranaike': 'CMB',
            'katunayake': 'CMB',
            'singapore': 'SIN',
            'changi': 'SIN',
            'dubai': 'DXB',
            'dxb': 'DXB',
            'mumbai': 'BOM',
            'bombay': 'BOM',
            'delhi': 'DEL',
            'chennai': 'MAA',
            'madras': 'MAA',
            'male': 'MLE',
            'malé': 'MLE',
            'bangkok': 'BKK',
            'suvarnabhumi': 'BKK',
            'kuala lumpur': 'KUL',
            'kul': 'KUL'
        }
        
        return airport_codes.get(location.lower().strip(), location.upper())

    def search_flights(self, query: str) -> Dict[str, Any]:
        """Search for flights based on a natural language query.
        
        Args:
            query: Natural language query (e.g., "flights from colombo to singapore tomorrow")
            
        Returns:
            Dict containing flight information or error message
        """
        if not self.api_key:
            return {"error": "Flight information is currently unavailable. Please try again later."}
            
        query = query.lower()
        
        try:
            # Extract departure and arrival locations
            dep = None
            arr = None
            
            if "from" in query and "to" in query:
                # Format: "flights from X to Y [date]"
                parts = query.split()
                from_idx = parts.index("from")
                to_idx = parts.index("to")
                
                # Get departure city (everything between "from" and "to")
                dep_parts = parts[from_idx + 1:to_idx]
                dep = " ".join(dep_parts)
                
                # Get arrival city (everything after "to" until the next keyword or end)
                date_keywords = ["today", "tomorrow", "on", "at", "for"]
                arr_parts = []
                for part in parts[to_idx + 1:]:
                    if part in date_keywords:
                        break
                    arr_parts.append(part)
                arr = " ".join(arr_parts)
                
                # Get date if specified
                date = None
                if "tomorrow" in query:
                    date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                elif "today" in query:
                    date = datetime.now().strftime("%Y-%m-%d")
                
                # Convert city names to IATA codes
                dep_code = self._get_airport_code(dep)
                arr_code = self._get_airport_code(arr)
                
                if dep_code and arr_code:
                    return self.get_flights_by_route(dep_code, arr_code, date)
                
            # If we couldn't parse the query, return a helpful error
            return {
                "error": "I couldn't understand your flight request. "
                        "Please try a format like: 'flights from colombo to singapore tomorrow'"
            }
            
        except Exception as e:
            return {"error": f"Error processing flight request: {str(e)}"}

# Singleton instance
_flight_service = None

def get_flight_service(api_key: str = None) -> FlightService:
    """Get or create the flight service instance."""
    global _flight_service
    if _flight_service is None:
        _flight_service = FlightService(api_key)
    return _flight_service
