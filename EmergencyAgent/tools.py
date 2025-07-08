from crewai_tools import PDFSearchTool
from dotenv import load_dotenv

load_dotenv()

first_aid_tool=PDFSearchTool(pdf=r"EmergencyAgent\FA-manual.pdf")