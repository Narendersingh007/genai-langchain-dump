from fastapi import FastAPI,Path , HTTPException, Query
import json
app = FastAPI()

def load_data():
    with open("patients.json","r") as f:
        data = json.load(f)
    return data


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
def sort_patients(sort_by : str = Query(...,description="Sort on the basis of height ,weight, bmi "),order : str = Query("asc",description="Order of sorting (asc or desc)")):
    valid_fields = ["height","weight","bmi"]
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f"Invalid sort_by value. Must be one of {valid_fields}")
    if order not in ["asc","desc"]:
        raise HTTPException(status_code=400, detail="Invalid order value. Must be 'asc' or 'desc'")
    data = load_data()
    sorted_order = True if order == "desc" else False
    sorted_patients = sorted(data.values(), key=lambda x: x.get(sort_by), reverse=sorted_order)
    return sorted_patients