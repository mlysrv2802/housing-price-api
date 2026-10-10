from pydantic import BaseModel

class NewListingsResponse(BaseModel):
    count:int
    listings:list