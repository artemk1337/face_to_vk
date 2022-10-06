from settings import VK_SESSION, VK_TOOLS, ITER_MAX_BUFFER, PHOTO_MAX_SHIFT_TIME, PARSER_ALL


class CityTransformer:
    METHOD_COUNTRY = "database.getCities"

    def __init__(self, country_id: int, q: str):
        self.country_id = country_id
        self.q = q
        self.cities_with_id = None

    def get_all_cities_with_id(self) -> list:
        self.cities_with_id: list = PARSER_ALL(
            self.METHOD_COUNTRY,
            1000,
            values={"country_id": self.country_id, "q": self.q},
        )['items']
        return self.cities_with_id

    def name2id(self):
        if not self.cities_with_id:
            self.get_all_cities_with_id()
        return self.cities_with_id[0]['id']


class CountryTransformer:
    METHOD_COUNTRY = "database.getCountries"

    def __init__(self):
        self.countries_dict = None

    def get_all_countries_name_id(self):
        self.countries_dict = {}
        countries: list = PARSER_ALL(
            self.METHOD_COUNTRY,
            1000,
        )['items']
        for country in countries:
            self.countries_dict[country['title']] = country['id']
        return self.countries_dict

    def name2id(self, name):
        if not self.countries_dict:
            self.get_all_countries_name_id()
        return self.countries_dict[name]
