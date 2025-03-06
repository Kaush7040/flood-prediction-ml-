import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 
import scipy 
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import openrouteservice as ors
import folium
import operator
from functools import reduce
import math
from IPython.display import display
import io
from PIL import Image
from streamlit_folium import st_folium
import time
import vonage
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

api_key = '' 

llm = GooglePalm(google_api_key=api_key, temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

api_key = '4f448163'
api_secret = 'P24TNhGJfYBJJIqO'

client = vonage.Client(key=api_key, secret=api_secret)
sms = vonage.Sms(client)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Based on the provided context and question, create a response strictly using the information from the "response" section of the source document.
    Use as much of the text from this section as possible with minimal modifications.
    If the context does not contain the answer, simply state "Refer to Kerala WRIS for further details". Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def send_sms(to, text):
    responseData = sms.send_message(
        {
            "from": "VonageAPI",
            "to": to,
            "text": text,
        }
    )

    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")



client = ors.Client(key='5b3ce3597851110001cf624876a2839afd6741a7992073ce7a614c8e')
x=pd.read_csv(r"C:\Code\disaster prediction project\Flood-prediction-master\approach1\kerala.csv")
rf_names = ["Kerala State Disaster Management Authority","HelpAge India","Welfare Services Ernakulam","ENNAKKATHARA AYURVEDA HOSPITAL","NEUROBASE BRAIN AND SPINE EXCELLENCE SPECIALITY CENTRE FOR AYURVEDA AND PHYSIOTHERAPY","Kripa Ayurveda Marma Hospital","Ayurmadom Ayurveda Treatment Centre"]
rf_coords = [
    [8.639980372316959, 76.95590935165335],
    [10.119520410385372, 76.29123648841443],
    [9.987917841891562, 76.31416575113498],
    [9.986565361210653, 76.31347910565768],
    [10.024094610996531, 76.32583872424894],
    [9.979802873495828, 76.2437845897126],
    [9.991299018994322, 76.29665629146405]
]

flood=[]
june=[]
sub=[]
y1=list(x["YEAR"])
x1=list(x["Jun-Sep"])
z1=list(x["JUN"])
w1=list(x["MAY"])
for i in range(0,len(x1)):
    if x1[i]>2400:
        flood.append('1')
    else:
        flood.append('0')
for k in range(0,len(x1)):
    june.append(z1[k]/3)
for k in range(0,len(x1)):
    average = (w1[k]+z1[k])/2
    difference = abs(w1[k]-z1[k])
    sub.append((difference / average) * 100)
df = pd.DataFrame({'flood':flood})
df1=pd.DataFrame({'per_10_days':june})
x["flood"]=flood
x["avgjune"]=june
x["sub"]=sub
x.to_csv("out1.csv")
X = x.iloc[:,[16,20,21]].values
y1=x.iloc[:,19].values
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y1, random_state=0)
Lr=LogisticRegression()
Lr.fit(X,y1)


    

def predict(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    time.sleep(5)  
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    summary_index = text.find("Summary")
    if summary_index != -1:
        summary_section = text[summary_index:]
        lines = summary_section.split('\n')
        actual_str = lines[1].strip().replace(",", "")
        normal_str = lines[2].strip().replace(",", "")
        deviation_str = lines[3].strip().replace(",", "")
        actual = float(actual_str)
        normal = float(normal_str)
        deviation = float(deviation_str)
        print(f"Actual: {actual}")
        print(f"Normal: {normal}")
        print(f"Deviation: {deviation}")
        st.subheader("Rainfall Data:")
        st.text("Actual Rainfall (mm):")
        st.write(actual)
        st.text("Normal Rainfall (mm):")
        st.write(normal)
        st.text("Deviation in Rainfall (%):")
        st.write(deviation)
    driver.quit()
    l=[[normal,actual,deviation]]
    f1=Lr.predict(l)
    for i in range(len(f1)):
        if (int(f1[i])==1):
            st.subheader("PREDICTION:")
            st.text("WARNING - POSSIBILITY OF FLOOD IN THIS DISTRICT")
            st.text("Warning message has been forwarded to your phone")
            recipient_phone_number = '917080942087' 
            message = 'Flood alert: Please evacuate immediately to higher ground. Follow this link for the nearest relief center:'
            send_sms(recipient_phone_number, message)
        else:
            st.subheader("PREDICTION:")
            st.text("Flooding event is unlikely")
    



def map(xc, yc):
    def calculate_distance(coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    user_coords = [xc, yc]
    user_coords = user_coords[::-1]
    
    for i in range(len(rf_coords)):
        rf_coords[i] = rf_coords[i][::-1]
    
    m = folium.Map(location=[12.357949812626646, 75.10041316496783], tiles="cartodbpositron", zoom_start=13)
    closest_coords = min(rf_coords, key=lambda coord: calculate_distance(user_coords, coord))

    st.subheader("The closest relief centre coordinates are:")
    st.text(closest_coords[::-1])
    st.subheader("Please go to centre name:")
    st.text(rf_names[rf_coords.index(closest_coords)])
    

    coords = [user_coords, closest_coords]
    route = client.directions(coordinates=coords, profile='foot-walking', format='geojson')

    waypoints = list(dict.fromkeys(reduce(operator.concat, [step['way_points'] for step in route['features'][0]['properties']['segments'][0]['steps']])))
    polyline_coords = [list(reversed(coord)) for coord in route['features'][0]['geometry']['coordinates']]

    folium.Marker(
        location=user_coords[::-1],
        popup="You",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)

    folium.Marker(
        location=closest_coords[::-1],
        popup=rf_names[rf_coords.index(closest_coords)],
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    folium.PolyLine(locations=polyline_coords, color="blue").add_to(m)

    st_folium(m,width=725)
    time.sleep(60)




st.title("Flood Prediction & Mapping System in Kerala")
nav = st.sidebar.radio("Navigation",["Flood Prediction","Relief Centres","FAQ Chatbot","About"])
if nav == "Flood Prediction":
    st.subheader("Flood Prediction System:")
    graph = st.selectbox("Choose between Kerala's 14 districts",["Click here","Kasaragod","Kannur","Kozhikode","Wayanad","Malappuram","Thrissur","Palakkad","Ernakulam","Alappuzha","Kottayam","Idukki","Pathanamthitta","Kollam","Thiruvananthapuram"])

    if graph == "Click here":
        pass
    if graph == "Kasaragod":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Kasaragod;loctype=DISTRICT;locuuid=4a0e14b2-de3e-4a7b-b1a4-a39e66ddbe48;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Kannur":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Kannur;loctype=DISTRICT;locuuid=02122ce9-dea4-4840-8745-5d3743e087a7;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Kozhikode":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Kozhikode;loctype=DISTRICT;locuuid=8189474d-71b9-488c-b45a-34bbcda95308;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Wayanad":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Wayanad;loctype=DISTRICT;locuuid=9638dab0-6fbe-4b7c-9f13-8908cc768221;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Malappuram":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Malappuram;loctype=DISTRICT;locuuid=10c9393d-79e0-4670-816a-8a05b395b13e;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Thrissur":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Thrissur;loctype=DISTRICT;locuuid=f81337dc-d04d-4618-9714-3609fe891919;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Palakkad":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Palakkad;loctype=DISTRICT;locuuid=1270f554-20cc-43ee-803e-1532f00e047c;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Ernakulam":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Ernakulam;loctype=DISTRICT;locuuid=3915d00f-2e7d-47e0-9113-f3824b61daae;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Alappuzha":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Alappuzha;loctype=DISTRICT;locuuid=46d05795-d880-42fb-824a-4c9ec61034a2;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Kottayam":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Kottayam;loctype=DISTRICT;locuuid=93bf1cb2-38d6-404c-8c21-a7beeec5c644;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Idukki":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Idukki;loctype=DISTRICT;locuuid=97ace365-2913-4b3d-8682-e796acfdc689;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Pathanamthitta":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Pathanamthitta;loctype=DISTRICT;locuuid=273d765a-fc5f-417a-bee8-a4257f271ca6;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")
    if graph == "Kollam":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Kollam;loctype=DISTRICT;locuuid=19bf47d8-4ba3-42ff-9dea-f1581c9d7ab7;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=true;handleEventFromBreadcrumb=true")
    if graph == "Thiruvananthapuram":
        predict("https://wris.kerala.gov.in/gis/rainfall;components=rainfall;srcName=IMD%20GRID;srcUUID=3d712275-a413-11ea-81b1-000d3a320689;minDate=19880701;maxDate=20240722;cType=MANDAL;locname=Thiruvananthapuram;loctype=DISTRICT;locuuid=787d19b3-b511-4161-b08c-aa23082318be;mapOnClickParams=true;component=rainfall;view=ADMIN;src=IMD%20GRID;type=ACTUAL;aggr=SUM;format=yyyyMMdd;ytd=2024;sDate=20240601;eDate=20240722;infotabOnClick=false;handleEventFromBreadcrumb=true")

    
if nav == "Relief Centres":
    st.subheader("Relief Centre Mapping System:")
    st.text("Find a relief centre closest to you from a list of flood relief centres")
    xc = st.number_input("Enter x coordinates",0.00,1000000.0000000)
    yc = st.number_input("Enter y coordinates",0.00,1000000.0000000)

    if st.button("Submit"):
        map(xc,yc)

        
if nav == "FAQ Chatbot":

    st.title("FAQ Chatbot")
    st.write("Ask your questions to our chatbot:")
    question = st.text_input("Question: ")

    if question:
        chain = get_qa_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])

if nav == "About":
    st.header("About this service")
    x=pd.read_csv(r"C:\Code\disaster prediction project\Flood-prediction-master\approach1\kerala.csv")
    y=pd.read_csv(r"C:\Code\disaster prediction project\Flood-prediction-master\approach1\kerala.csv")

    y1=list(x["YEAR"])
    x1=list(x["Jun-Sep"])
    z1=list(x["JUN"])
    w1=list(x["MAY"])

    plt.plot(y1, x1,'*')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("Welcome to the Flood Prediction System for the state of Kerala!")
    st.write("We use machine learning with the historical data of rainfall patterns to predict floods")
    st.write("If a flood is predicted, you are notified with a SMS warning sent to your phone no.")
    st.write("You can also use our Mapping System to generate a map to the closest relief centre from your location")
    
    st.pyplot()
    st.write("This is a scatter plot of the historical rainfall data we have used to train our ML model to predict floods")
