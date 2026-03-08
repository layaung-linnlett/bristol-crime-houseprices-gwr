import pytest

@pytest.fixture
def sample_house_prices():
    return [
        {'price': 250000, 'bedrooms': 3, 'location': 'Location A'},
        {'price': 300000, 'bedrooms': 4, 'location': 'Location B'},
        {'price': 150000, 'bedrooms': 2, 'location': 'Location C'}
    ]

@pytest.fixture
def sample_crime_data():
    return [
        {'crime_type': 'burglary', 'count': 5, 'location': 'Location A'},
        {'crime_type': 'assault', 'count': 2, 'location': 'Location B'},
        {'crime_type': 'theft', 'count': 8, 'location': 'Location C'}
    ]

@pytest.fixture
def sample_geospatial_data():
    return [
        {'location': 'Location A', 'coordinates': (51.5072, -0.1276)},
        {'location': 'Location B', 'coordinates': (51.5155, -0.0922)},
        {'location': 'Location C', 'coordinates': (51.5115, -0.1198)}
    ]
