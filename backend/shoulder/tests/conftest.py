import pytest  
  
from rest_framework.test import APIClient  
  
  
@pytest.fixture(scope="function")  
def api_client() -> APIClient:  
    """  
    Fixture to provide an API client  
    
    returns: APIClient  
    """  
    yield APIClient()
