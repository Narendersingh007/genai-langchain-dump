from pydantic import BaseModel, Field ,EmailStr
from typing import Optional
class Student(BaseModel):
    name : str = 'nitish'
    age : Optional[int] = None
    email : Optional[EmailStr] = None
    cgpa : float  = Field(gt = 0 , lt = 10,default =5 , description="CGPA must be between 0 and 10")

new_student = {'age' : '25','email':'abc@gmail.com'}
student = Student(**new_student)
print(type(student))
print(student)
student_dict = dict(student)
print(student_dict['age'])
student_json = student.model_dump_json()
print(student_json)