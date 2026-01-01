from typing import TypedDict, Optional

class Person(TypedDict):
    name : str
    age : int 

new_person : Person= {"name":"joy" , "age": 34}
print(new_person)
