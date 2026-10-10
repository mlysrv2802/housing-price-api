from pydantic import BaseModel,Field

class HouseFeatures(BaseModel):
    area_sqft:float = Field(..., gt=0, description="Area of house in square feet")
    bedrooms:int = Field(..., ge=0, le=20)
    bathrooms:int = Field(..., ge=0, le=20)
    age_years:int = Field(..., ge=0, le=200)
    distance_to_city_km:float = Field(..., ge=0)
    location_tier:str = Field(..., description="One of: Tier1, Tier2, Tier3")
   
    #metadata for OpenAPI/Swagger sample request body
    class Config:
        json_schema_extra = {
            "example":{
              "area_sqft":1400,
              "bedrooms":3,
              "bathrooms":2,
              "age_years":5,
              "distance_to_city_km":6.2,
              "location_tier":"Tier2",
            }
        }
