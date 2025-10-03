from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from gradio_client import Client
import re

# Enum for gender
class Gender(str, Enum):
    male = "Male"
    female = "Female"

app = FastAPI(title="Biomarker Health API", version="1.0")

client = Client("Muhammadidrees/MoizMedgemma27b")

class BiomarkerInput(BaseModel):
    albumin: float = Field(..., example=4.5, description="g/dL")
    creatinine: float = Field(..., example=1.5, description="mg/dL")
    glucose: float = Field(..., example=160, description="mg/dL")
    crp: float = Field(..., example=2.5, description="mg/L")
    mcv: float = Field(..., example=150, description="fL")
    rdw: float = Field(..., example=15, description="%")
    alp: float = Field(..., example=146, description="U/L")
    wbc: float = Field(..., example=10.5, description="x10^9/L")
    lymphocytes: float = Field(..., example=38, description="%")
    age: int = Field(..., example=30, description="Years")
    gender: Gender = Field(..., description="Select 'Male' or 'Female'")
    height: float = Field(..., example=123, description="cm")
    weight: float = Field(..., example=60, description="kg")

def parse_result_to_json(result_text: str):
    """
    Converts LLM markdown output into structured JSON
    """
    data = {
        "normal_ranges": {},
        "biomarker_table": [],
        "executive_summary": {"top_priorities": [], "key_strengths": []},
        "system_analysis": "",
        "action_plan": {"nutrition": "", "lifestyle": "", "medical": "", "testing": ""},
        "interaction_alerts": []
    }

    # Extract normal ranges
    normal_ranges = re.findall(r"- ([A-Za-z ]+): ([0-9.\-–]+.*)", result_text)
    for biomarker, value in normal_ranges:
        data["normal_ranges"][biomarker.strip()] = value.strip()

    # Extract biomarker table
    table_match = re.search(r"\| Biomarker \| Value \|.*?\|\n((?:\|.*\|\n)+)", result_text, re.S)
    if table_match:
        rows = table_match.group(1).strip().split("\n")
        for row in rows:
            parts = [p.strip() for p in row.strip("|").split("|")]
            if len(parts) == 4:
                data["biomarker_table"].append({
                    "biomarker": parts[0],
                    "value": parts[1],
                    "status": parts[2],
                    "insight": parts[3],
                })

    # Executive summary (priorities + strengths)
    priorities = re.findall(r"\d+\.\s+(.*)", result_text)
    data["executive_summary"]["top_priorities"] = priorities[:3]
    strengths = re.findall(r"- Normal (.*)", result_text)
    data["executive_summary"]["key_strengths"] = strengths

    # System analysis
    sys_match = re.search(r"System-Specific Analysis\n- Status: (.*?)\n- Explanation: (.*?)\n", result_text, re.S)
    if sys_match:
        data["system_analysis"] = {
            "status": sys_match.group(1).strip(),
            "explanation": sys_match.group(2).strip()
        }

    # Action plan
    plan_matches = re.findall(r"- (\w+): (.*)", result_text)
    for category, content in plan_matches:
        key = category.lower()
        if key in data["action_plan"]:
            data["action_plan"][key] = content.strip()

    # Interaction alerts
    interactions = re.findall(r"- (The .*?)\n", result_text)
    data["interaction_alerts"] = interactions

    return data

@app.post("/analyze")
async def analyze_biomarkers(data: BiomarkerInput):
    try:
        # Step 1: Get LLM result
        result = client.predict(
            albumin=data.albumin,
            creatinine=data.creatinine,
            glucose=data.glucose,
            crp=data.crp,
            mcv=data.mcv,
            rdw=data.rdw,
            alp=data.alp,
            wbc=data.wbc,
            lymphocytes=data.lymphocytes,
            age=data.age,
            gender=data.gender.value,
            height=data.height,
            weight=data.weight,
            api_name="/respond"
        )

        # Step 2: Parse markdown into JSON
        parsed = parse_result_to_json(result)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))