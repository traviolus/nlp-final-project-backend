from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum


class PageEnum(str, Enum):
    t047 = 't047'
    dinosaturday = 'dinosaturday'
    quotes = 'quotes'


default_router = APIRouter(
    prefix='/api',
    tags=['bertong'],
    responses={404: {'description': 'Not Found'}},
)


@default_router.get('/sentence/')
def get_sentence(start_text: str, page: PageEnum):
    return JSONResponse(content={'message': 'OK'})
