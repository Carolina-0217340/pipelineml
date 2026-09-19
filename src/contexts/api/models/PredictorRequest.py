from pydantic import BaseModel, validator


class PredictorRequest(BaseModel):
    email: str
    country: str
    city: str

    @validator("email")
    def validate_email(cls, email):
        if "@" not in email:
            raise ValueError("email must be a valid email address")
        return email

    @validator("country")
    def validate_country(cls, country):
        if not country or not isinstance(country, str):
            raise ValueError("country must be a non-empty string")
        return country

    @validator("city")
    def validate_city(cls, city):
        if not city or not isinstance(city, str):
            raise ValueError("city must be a non-empty string")
        return city