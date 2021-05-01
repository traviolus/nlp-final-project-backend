from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from random import uniform

from models import DinoModel_obj, T047Model_obj, MixModel_obj, QuoteModel_obj, DataTools

class PageEnum(str, Enum):
    t047 = 't047'
    dinosaturday = 'dinosaturday'
    mix = 'mix'
    quotes = 'quotes'


default_router = APIRouter(
    prefix='/api',
    tags=['bertong'],
    responses={404: {'description': 'Not Found'}},
)


@default_router.get('/sentence/')
def get_sentence(start_text: str, page: PageEnum):
    try:
        if page == 'quotes':
            generated_text = DataTools.gen_sent(start_text, QuoteModel_obj)
        else:
            model_dict = {'dinosaturday': DinoModel_obj, 't047': T047Model_obj, 'mix': MixModel_obj}
            temperature = round(uniform(1,2), 1)
            generated_text = DataTools.temperature_sampling_decode(start_text, 100, model_dict[page], temperature)
        return JSONResponse(content={'message': generated_text})
    except KeyError:
        return JSONResponse(content={'message': 'คำเริ่มต้นไม่ถูกต้อง กรุณาลองใหม่'}, status_code=status.HTTP_400_BAD_REQUEST)
