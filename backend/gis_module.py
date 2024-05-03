# Module for common GIS utilities relevant to shoulder2shoulder.
from geopy.geocoders import Nominatim
from geopy import distance

def geocode(address):
    '''
    Geocode an address as a string and returns dictionarity with the standardized address and a 
    returns a 'coords' key with a tuple that has latitude, then longitude.
    '''

    # Nominatim has a rate limit of 1 second. For testing purposes, 
    # I am using my (Ethan's) email as the registered user.
    geolocator = Nominatim(user_agent = "ethanarsht@gmail.com")
    location = geolocator.geocode(address)

    return {'address': location.address, 'coords': (location.latitude, location.longitude)}

def distance_bin(loc_one, loc_two, ellipsoid = "WGS-84", unit = "miles"):
    '''
    Takes two coordinate sets (tuples) and returns a distance in miles and a distance bin, using this logic:
    0 - 5 miles difference: 0
    5 - 10 miles : 1
    10 - 15 miles: 2
    Returns a tuple of format (distance, bin)
    ...
    '''

    dist = eval(f'distance.distance(loc_one, loc_two, ellipsoid = ellipsoid).{unit}')
    bin = dist // 5

    return (dist, bin)


def travel_time(coord_one, coord_two, mode = 'car'):
    '''
    Takes in two coordinates and calculates travel time. Returns an integer in seconds of travel time.
    Not implemented.
    '''


    

    
    



