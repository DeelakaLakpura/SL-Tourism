"""
IATA Code Lookup Utility

This module provides functionality to convert between city names, airport names,
and IATA codes using the AviationStack API and a local cache of common airports.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass

@dataclass
class Airport:
    """Data class representing an airport."""
    iata_code: str
    icao_code: str
    name: str
    city: str
    country: str
    lat: float
    lng: float

class IATACodeLookup:
    """Handles lookup and conversion of IATA codes for airports."""
    
    def __init__(self, api_key: str = None):
        """Initialize with optional API key.
        
        Args:
            api_key: AviationStack API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY")
        self.airports = self._load_common_airports()
    
    def _load_common_airports(self) -> Dict[str, Airport]:
        """Load a static list of common airports as a fallback."""
        common_airports = {
          'DXB': Airport('DXB', 'OMDB', 'Dubai International Airport', 'Dubai', 'United Arab Emirates', 25.2532, 55.3657),
    'HND': Airport('HND', 'RJTT', 'Tokyo Haneda Airport', 'Tokyo', 'Japan', 35.5494, 139.7798),
    'LAX': Airport('LAX', 'KLAX', 'Los Angeles International Airport', 'Los Angeles', 'United States', 33.9416, -118.4085),
    'CDG': Airport('CDG', 'LFPG', 'Charles de Gaulle Airport', 'Paris', 'France', 49.0097, 2.5479),
    'AMS': Airport('AMS', 'EHAM', 'Amsterdam Schiphol Airport', 'Amsterdam', 'Netherlands', 52.3105, 4.7683),
    'FRA': Airport('FRA', 'EDDF', 'Frankfurt am Main Airport', 'Frankfurt', 'Germany', 50.1109, 8.6821),
    'ICN': Airport('ICN', 'RKSI', 'Incheon International Airport', 'Seoul', 'South Korea', 37.4602, 126.4407),
    'SYD': Airport('SYD', 'YSSY', 'Sydney Kingsford Smith Airport', 'Sydney', 'Australia', -33.9399, 151.1753),
    'YYZ': Airport('YYZ', 'CYYZ', 'Toronto Pearson International Airport', 'Toronto', 'Canada', 43.6777, -79.6248),
    'GRU': Airport('GRU', 'SBGR', 'São Paulo/Guarulhos International Airport', 'São Paulo', 'Brazil', -23.4356, -46.4731),
    'JNB': Airport('JNB', 'FAOR', 'O. R. Tambo International Airport', 'Johannesburg', 'South Africa', -26.1392, 28.2460),
    'MIA': Airport('MIA', 'KMIA', 'Miami International Airport', 'Miami', 'United States', 25.7959, -80.2871),
    'MAD': Airport('MAD', 'LEMD', 'Adolfo Suárez Madrid–Barajas Airport', 'Madrid', 'Spain', 40.4983, -3.5676),
    'BCN': Airport('BCN', 'LEBL', 'Barcelona–El Prat Airport', 'Barcelona', 'Spain', 41.2974, 2.0833),
    'MEX': Airport('MEX', 'MMMX', 'Mexico City International Airport', 'Mexico City', 'Mexico', 19.4361, -99.0719),
    'KUL': Airport('KUL', 'WMKK', 'Kuala Lumpur International Airport', 'Kuala Lumpur', 'Malaysia', 2.7456, 101.7072),
    'ZRH': Airport('ZRH', 'LSZH', 'Zurich Airport', 'Zurich', 'Switzerland', 47.4647, 8.5492),
    'VIE': Airport('VIE', 'LOWW', 'Vienna International Airport', 'Vienna', 'Austria', 48.1103, 16.5697),
    'IST': Airport('IST', 'LTFM', 'Istanbul Airport', 'Istanbul', 'Turkey', 41.2753, 28.7519),
    'DME': Airport('DME', 'UUDD', 'Domodedovo International Airport', 'Moscow', 'Russia', 55.4088, 37.9063),
    'DOH': Airport('DOH', 'OTHH', 'Hamad International Airport', 'Doha', 'Qatar', 25.2736, 51.6080),
    'BKK': Airport('BKK', 'VTBS', 'Suvarnabhumi Airport', 'Bangkok', 'Thailand', 13.6900, 100.7501),
    'HKG': Airport('HKG', 'VHHH', 'Hong Kong International Airport', 'Hong Kong', 'China', 22.3080, 113.9185),
    'MUC': Airport('MUC', 'EDDM', 'Munich Airport', 'Munich', 'Germany', 48.3538, 11.7861),
    'SVO': Airport('SVO', 'UUEE', 'Sheremetyevo International Airport', 'Moscow', 'Russia', 55.9726, 37.4146),
    'ATH': Airport('ATH', 'LGAV', 'Athens International Airport', 'Athens', 'Greece', 37.9364, 23.9475),
    'OSL': Airport('OSL', 'ENGM', 'Oslo Gardermoen Airport', 'Oslo', 'Norway', 60.1939, 11.1004),
    'ARN': Airport('ARN', 'ESSA', 'Stockholm Arlanda Airport', 'Stockholm', 'Sweden', 59.6519, 17.9186),
    'CPH': Airport('CPH', 'EKCH', 'Copenhagen Airport', 'Copenhagen', 'Denmark', 55.6180, 12.6560),
    'HEL': Airport('HEL', 'EFHK', 'Helsinki-Vantaa Airport', 'Helsinki', 'Finland', 60.3172, 24.9633),
    'BRU': Airport('BRU', 'EBBR', 'Brussels Airport', 'Brussels', 'Belgium', 50.9010, 4.4844),
    'DUB': Airport('DUB', 'EIDW', 'Dublin Airport', 'Dublin', 'Ireland', 53.4213, -6.2701),
    'PRG': Airport('PRG', 'LKPR', 'Václav Havel Airport Prague', 'Prague', 'Czech Republic', 50.1008, 14.2632),
    'WAW': Airport('WAW', 'EPWA', 'Warsaw Chopin Airport', 'Warsaw', 'Poland', 52.1657, 20.9671),
    'BUD': Airport('BUD', 'LHBP', 'Budapest Ferenc Liszt International Airport', 'Budapest', 'Hungary', 47.4369, 19.2556),
    'LIS': Airport('LIS', 'LPPT', 'Humberto Delgado Airport', 'Lisbon', 'Portugal', 38.7742, -9.1342),
    'SCL': Airport('SCL', 'SCEL', 'Arturo Merino Benítez Airport', 'Santiago', 'Chile', -33.3930, -70.7858),
    'BOG': Airport('BOG', 'SKBO', 'El Dorado International Airport', 'Bogotá', 'Colombia', 4.7016, -74.1469),
    'EZE': Airport('EZE', 'SAEZ', 'Ministro Pistarini International Airport', 'Buenos Aires', 'Argentina', -34.8222, -58.5358),
    'LIM': Airport('LIM', 'SPJC', 'Jorge Chávez International Airport', 'Lima', 'Peru', -12.0219, -77.1143),
    'NRT': Airport('NRT', 'RJAA', 'Narita International Airport', 'Tokyo', 'Japan', 35.7765, 140.3189),
    'TPE': Airport('TPE', 'RCTP', 'Taiwan Taoyuan International Airport', 'Taipei', 'Taiwan', 25.0777, 121.2328),
    'SFO': Airport('SFO', 'KSFO', 'San Francisco International Airport', 'San Francisco', 'United States', 37.6213, -122.3790),
    'SEA': Airport('SEA', 'KSEA', 'Seattle-Tacoma International Airport', 'Seattle', 'United States', 47.4502, -122.3088),
    'ORD': Airport('ORD', 'KORD', 'O\'Hare International Airport', 'Chicago', 'United States', 41.9742, -87.9073),
    'BOM': Airport('BOM', 'VABB', 'Chhatrapati Shivaji Maharaj International Airport', 'Mumbai', 'India', 19.0896, 72.8656),
    'BLR': Airport('BLR', 'VOBL', 'Kempegowda International Airport', 'Bangalore', 'India', 13.1986, 77.7066),
    'MAA': Airport('MAA', 'VOMM', 'Chennai International Airport', 'Chennai', 'India', 12.9941, 80.1709),
    'CGK': Airport('CGK', 'WIII', 'Soekarno–Hatta International Airport', 'Jakarta', 'Indonesia', -6.1256, 106.6558),
        }
        return common_airports
    
    def search_airport(self, query: str) -> Optional[Airport]:
        """Search for an airport by IATA code, name, city, or country.
        
        Args:
            query: Search term (IATA code, airport name, city, or country)
            
        Returns:
            Matching Airport object or None if not found
        """
        if not query:
            return None
            
        query = query.upper().strip()
        
        # First check direct IATA code match
        if query in self.airports:
            return self.airports[query]
            
        # Then search through airport details
        for airport in self.airports.values():
            if (query == airport.city.upper() or 
                query in airport.name.upper() or 
                query == airport.country.upper()):
                return airport
                
        # If not found in local cache, try API lookup
        if self.api_key:
            return self._search_airport_via_api(query)
            
        return None
    
    def _search_airport_via_api(self, query: str) -> Optional[Airport]:
        """Search for an airport using the AviationStack API."""
        try:
            params = {
                'access_key': self.api_key,
                'search': query,
                'limit': 1
            }
            
            response = requests.get(
                'http://api.aviationstack.com/v1/airports',
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    airport_data = data['data'][0]
                    return Airport(
                        iata_code=airport_data.get('iata_code'),
                        icao_code=airport_data.get('icao_code'),
                        name=airport_data.get('airport_name'),
                        city=airport_data.get('location_name'),
                        country=airport_data.get('country_name'),
                        lat=float(airport_data.get('latitude', 0)),
                        lng=float(airport_data.get('longitude', 0))
                    )
        except Exception as e:
            print(f"Error searching airport via API: {e}")
            
        return None
    
    def get_iata_code(self, location: str) -> Optional[str]:
        """Get IATA code for a location (city, airport name, or country)."""
        airport = self.search_airport(location)
        return airport.iata_code if airport else None
    
    def get_airport_info(self, location: str) -> Optional[Dict]:
        """Get detailed airport information for a location."""
        airport = self.search_airport(location)
        if airport:
            return {
                'iata_code': airport.iata_code,
                'name': airport.name,
                'city': airport.city,
                'country': airport.country,
                'coordinates': {
                    'lat': airport.lat,
                    'lng': airport.lng
                }
            }
        return None

# Singleton instance
_iata_lookup = None

def get_iata_lookup(api_key: str = None) -> IATACodeLookup:
    """Get or create the IATA code lookup instance."""
    global _iata_lookup
    if _iata_lookup is None:
        _iata_lookup = IATACodeLookup(api_key)
    return _iata_lookup
