from pydantic import BaseModel
from typing import List, Optional


class ThumbsAction(BaseModel):
    question: str
    answer: str
    comment: Optional[str]
    
class QuestionInput(BaseModel):
    question: str

class AnswerOutput(BaseModel):
    content: str
    source: str
    source_document: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    final_answer: str
    answers: List[AnswerOutput]
    source_document: str
