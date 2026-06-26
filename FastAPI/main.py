from fastapi import FastAPI,Path , HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field , computed_field
from typing import Annotated , Literal , Optional
import json
app = FastAPI()

class Patient(BaseModel) :
    id : Annotated[str,Field(...,description = " ID of patient", examples = ['P001'])]
    name : Annotated[str,Field(...,description = " Name of patient")]
    city :Annotated[str,Field(...,description = " city where patient is living")]
    age : Annotated[int,Field(...,gt=0,lt=120,description = " Age of patient")]
    gender : Annotated[Literal['male','female','others'],Field(...,description = " Gender of patient")]
    height : Annotated[float,Field(...,gt = 0 , description = "Height of patient in metres")]
    weight : Annotated[float,Field(...,gt = 0 , description = "Weight of patient in kgs")]
    @computed_field
    @property
    def bmi(self)->float :
        bmi = round(self.weight / (self.height ** 2),2)
        return bmi
    
    @computed_field
    @property
    def verdict(self)->str:
        if self.bmi<18.5 :
            return 'Underweight'
        elif self.bmi<25:
            return 'Normal'
        elif self.bmi <30 :
            return 'Normal'
        else :
            return 'Obese'

class PatientUpdate(BaseModel):
    name : Annotated[Optional[str],Field(default = None)]
    city :Annotated[Optional[str],Field(default = None)]
    age : Annotated[Optional[int],Field(default = None)]
    gender : Annotated[Optional[Literal['male','female','others']],Field(default = None)]
    height : Annotated[Optional[float],Field(default = None)]
    weight : Annotated[Optional[float],Field(default = None)]

def load_data():
    with open("patients.json","r") as f:
        data = json.load(f)
    return data
def save_data(data):
    with open("patients.json","w") as f:
          json.dump(data,f)

@app.get("/")
def hello():
    return {"message": "Patient Management System API"}

@app.get("/about")
def about():
    return {"message": "This is a simple Patient Management System API built with FastAPI."}

@app.get("/view")
def view_patients():
    data = load_data()
    return data

@app.get("/patient/{patient_id}")
def view_patient(patient_id : str = Path(..., description = "The ID of the patient to retrieve",example = "P001")):
    data = load_data()
    if patient_id in data : 
        return data[patient_id]
    raise HTTPException(status_code =404 , detail = "patient not found")

@app.get("/sort")
def sort_patients(sort_by : str = Query(...,description = "Sort on the basis og height,weight,bmi"),order : str = Query("asc",description ="Oruvicornder of sorting (asc or desc)" )):
    valid_fields = ["height","weoght","bmi"]
    if sort_by not in valid_fields :
        raise HTTPException(status_code = 400 , detail = f"Invalid sort_by value . Must be one of {valid_fields}")
    if order not in ["asc","desc"]:
        raise HTTPException(status_code=400, detail="Invalid order value. Must be 'asc' or 'desc'")
    data = load_data()
    sorted_order = True if order =="desc" else False
    sorted_patients = sorted(data.values(),key= lambda x : x.get(sort_by),reverse=sorted_order)
    return sorted_patients

@app.post('/create')
def create_patient(patient : Patient) :
    # load existing data 
    data = load_data()

     # check if patient already exist
    if patient.id in data :
        raise HTTPException(status_code = 400 , detail = ' Patient already exists')
        
    # new patient add to the database
    data[patient.id] = patient.model_dump(exclude = ['id'])

    ## save into json file 
    save_data(data)
    return JSONResponse(status_code = 201 , content = {'message':'patient created successfully'})

@app.put('/edit/{patient_id}')
def update_patient(patient_id : str , patient_update : PatientUpdate) :
    data = load_data()
    if patient_id not in data :
        raise HTTPException(status_code = 404 , detail = 'Patient not found')
    existing_patient_info = data[patient_id]
    updated_patient_info = patient_update.model_dump(exclude_unset = True)
    for key,value in updated_patient_info.items() :
        existing_patient_info[key] = value
    # making a new pydantic object with existing_patient_info to get computed fileds adjusted properly
    existing_patient_info['id'] = patient_id

    patient_pydantic_object = Patient(**existing_patient_info)
    existing_patient_info_patient_info = patient_pydantic_object.model_dump(exclude ='id')
    data[patient_id] = existing_patient_info
    save_data(data)
    return JSONResponse(status_code = 200,content = {'message':'patient updated'})

@app.delete("/delete/{patient_id}")
def delete_patient(patiend_id : str) :
    data = load_data()
    if patiend_id not in data :
        raise HTTPException(status_code = 404 , detail = 'Patient not found')
    data.pop(patiend_id,None)
    save_data(data)
    return JSONResponse(status_code = 200, content={'message':'patient deleted'})
